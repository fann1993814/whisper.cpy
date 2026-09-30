from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np


SpeechDetector = Callable[[np.ndarray], bool]


class BasePolicy(ABC):
    """
    Abstract policy for streaming ASR.

    A policy controls:

    - whether the current inference window should be processed
    - whether Whisper decoding should run
    - whether an utterance has ended
    - when the transcript should be flushed
    - streaming / inference timing state

    The policy does not own audio buffering.

    Audio buffering and inference-window construction are handled by
    the audio buffer.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the policy state."""
        raise NotImplementedError

    @abstractmethod
    def add_samples(self, n_samples: int) -> None:
        """
        Notify the policy that new audio samples have arrived.

        This is used to maintain the streaming timeline.
        """
        raise NotImplementedError

    @abstractmethod
    def should_process(self, window: np.ndarray) -> bool:
        """
        Whether the current audio window should be processed.

        Policy-level conditions such as VAD are evaluated here.
        """
        raise NotImplementedError

    @abstractmethod
    def should_decode(self) -> bool:
        """
        Whether Whisper inference should run for the current iteration.
        """
        raise NotImplementedError

    @abstractmethod
    def on_decode_start(self) -> None:
        """Notify the policy that Whisper decoding has started."""
        raise NotImplementedError

    @abstractmethod
    def on_iteration(self) -> None:
        """Notify the policy that one streaming iteration has completed."""
        raise NotImplementedError

    @abstractmethod
    def is_utterance_end(self) -> bool:
        """Whether the current utterance has reached end-of-utterance."""
        raise NotImplementedError

    @abstractmethod
    def should_flush(self, force: bool = False) -> bool:
        """Whether the current transcript should be flushed."""
        raise NotImplementedError

    @abstractmethod
    def on_flush(self) -> None:
        """Notify the policy that a flush has occurred."""
        raise NotImplementedError

    @property
    @abstractmethod
    def stream_ms(self) -> int:
        """Current stream position in milliseconds."""
        raise NotImplementedError


class FixedStepPolicy(BasePolicy):
    """
    Fixed-step streaming policy.

    Responsibilities:

    - speech detection
    - silence / EOU detection
    - continuous processing while an utterance is active
    - inference-time protection
    - periodic forced decoding
    - periodic transcript flushing

    The policy does not own audio data.

    The audio buffer determines when enough new audio is available
    to construct an inference window.
    """

    def __init__(
        self,
        step_samples: int,
        length_samples: int,
        sample_rate: int = 16000,
        speech_detector: Optional[SpeechDetector] = None,
        eou_duration_ms: int = 1000,
    ) -> None:
        if step_samples <= 0:
            raise ValueError(
                "step_samples must be positive"
            )

        if length_samples <= 0:
            raise ValueError(
                "length_samples must be positive"
            )

        if length_samples < step_samples:
            length_samples = step_samples

        if sample_rate <= 0:
            raise ValueError(
                "sample_rate must be positive"
            )

        if eou_duration_ms < 0:
            raise ValueError(
                "eou_duration_ms must be non-negative"
            )

        self.step_samples = step_samples
        self.length_samples = length_samples
        self.sample_rate = sample_rate

        self.speech_detector = speech_detector

        # ------------------------------------------------------------------
        # Speech state
        # ------------------------------------------------------------------

        # True after speech has been detected and until EOU / flush.
        self.active_speech = False

        # Number of consecutive non-speech samples after speech started.
        #
        # This is measured using NEW audio samples only. It must not be
        # derived from len(window), because the inference window contains
        # previously processed audio and may contain overlap.
        self.silence_samples = 0

        # Silence duration required to declare EOU.
        self.eou_samples = int(
            sample_rate * eou_duration_ms / 1000
        )

        # ------------------------------------------------------------------
        # Iteration state
        # ------------------------------------------------------------------

        # This is incremented after a processable streaming iteration,
        # regardless of whether Whisper inference actually runs.
        self.n_iter = 0

        self.n_new_line = max(
            1,
            int(length_samples / step_samples - 1),
        )

        # ------------------------------------------------------------------
        # Streaming timeline
        # ------------------------------------------------------------------

        # Total number of audio samples received by the stream.
        self.stream_samples = 0

        # Stream position observed by should_process().
        #
        # This is used to determine how many NEW samples have arrived
        # since the previous processable window.
        self.prev_process_stream_samples = 0

        # ------------------------------------------------------------------
        # Inference timing
        # ------------------------------------------------------------------

        # Stream position at which the previous Whisper inference started.
        self.prev_inference_start_stream_samples = 0

        # Wall-clock time at which the previous Whisper inference started.
        self.prev_inference_start_timing = 0.0

    # ----------------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset policy state.

        This corresponds to starting a new streaming session.
        """

        self.active_speech = False
        self.silence_samples = 0

        self.n_iter = 0

        self.stream_samples = 0
        self.prev_process_stream_samples = 0

        self.prev_inference_start_stream_samples = 0
        self.prev_inference_start_timing = 0.0

    # ----------------------------------------------------------------------
    # Streaming timeline
    # ----------------------------------------------------------------------

    def add_samples(self, n_samples: int) -> None:
        """
        Add newly received samples to the streaming timeline.

        This represents the global stream position and is independent
        of the current inference window.
        """

        if n_samples < 0:
            raise ValueError(
                "n_samples must be non-negative"
            )

        self.stream_samples += n_samples

    # ----------------------------------------------------------------------
    # Speech / EOU
    # ----------------------------------------------------------------------

    def should_process(self, window: np.ndarray) -> bool:
        """
        Decide whether the current inference window should be processed.

        The input is the complete sliding inference window constructed
        by the audio buffer.

        Before speech starts:
            - run speech detection on the window
            - wait until speech is detected

        After speech starts:
            - continue processing both speech and silence
            - accumulate newly received silence for EOU detection

        Important:

            `window` may contain previously processed audio and overlap.
            Therefore len(window) must NOT be used to measure streaming
            progress or silence duration.

            Silence duration is measured from the global streaming
            timeline using the number of NEW samples received since the
            previous processable window.
        """

        window = np.asarray(window)

        if window.ndim != 1:
            raise ValueError(
                "Policy expects a mono audio array."
            )

        if window.size == 0:
            return False

        # --------------------------------------------------------------
        # Determine how much NEW audio has arrived since the previous
        # processable window.
        #
        # Do not use len(window):
        #
        #     window = previous audio + overlap + new audio
        #
        # The stream timeline is the source of truth.
        # --------------------------------------------------------------

        new_samples = (
            self.stream_samples
            - self.prev_process_stream_samples
        )

        if new_samples < 0:
            raise RuntimeError(
                "Streaming timeline moved backwards."
            )

        self.prev_process_stream_samples = self.stream_samples

        # --------------------------------------------------------------
        # Speech detection
        # --------------------------------------------------------------

        if self.speech_detector is None:
            self.active_speech = True
            is_speech = True
        else:
            last_chunk = window[-new_samples*2:] if new_samples > 0 else window
            is_speech = self.speech_detector(last_chunk)

        # --------------------------------------------------------------
        # Speech detected
        # --------------------------------------------------------------

        if is_speech:
            self.active_speech = True
            self.silence_samples = 0

            return True

        # --------------------------------------------------------------
        # No speech
        # --------------------------------------------------------------

        if not self.active_speech:
            # No utterance has started yet.
            #
            # Keep waiting for speech. The audio buffer continues to
            # receive audio independently of this decision.
            return False

        # --------------------------------------------------------------
        # Speech has already started.
        #
        # Continue processing silence so that:
        #
        #     speech -> silence -> EOU
        #
        # can be detected.
        #
        # Only newly received samples are counted.
        # --------------------------------------------------------------

        self.silence_samples += new_samples

        return True

    def is_utterance_end(self) -> bool:
        """
        Return True when enough consecutive silence has accumulated
        after speech.

        EOU does not reset the policy state.

        The streaming ASR layer decides how to handle the EOU event,
        such as performing a final decode and committing the current
        transcript.
        """

        if not self.active_speech:
            return False

        return self.silence_samples >= self.eou_samples

    # ----------------------------------------------------------------------
    # Decode scheduling
    # ----------------------------------------------------------------------

    def should_decode(self) -> bool:
        """
        Decide whether Whisper inference should run.

        If there has been no previous inference, decode immediately.

        Otherwise, compare:

            inference wall-clock time

        against:

            amount of new audio accumulated since the previous
            inference started.

        This prevents Whisper inference from continuously falling
        behind the incoming audio stream.
        """

        # No previous inference.
        # was used to determine whether a previous inference existed.
        if self.prev_inference_start_timing <= 0:
            return True

        # How long the previous inference has been running.
        inference_spend_time = int(
            (
                time.time()
                - self.prev_inference_start_timing
            ) * 100
        ) / 100

        # How much new audio has arrived since the previous inference
        # started.
        inference_consume_audio_time = (
            self.stream_samples
            - self.prev_inference_start_stream_samples
        ) / self.sample_rate

        # Preserve the original scheduling rule exactly.
        return (
            inference_spend_time <= inference_consume_audio_time
            or self.is_flush_iteration()
        )

    def on_decode_start(self) -> None:
        """
        Record the beginning of a Whisper inference.

        The audio position is recorded at the same time as the wall-clock
        timestamp, matching the original implementation.
        """

        self.prev_inference_start_timing = time.time()

        self.prev_inference_start_stream_samples = (
            self.stream_samples
        )

    # ----------------------------------------------------------------------
    # Iteration / flush scheduling
    # ----------------------------------------------------------------------

    def on_iteration(self) -> None:
        """
        Mark one processable streaming iteration as completed.

        Important:

        This is called even when should_decode() returned False.

        which happened after the speech-processing block regardless of
        whether transcribe() was actually called.
        """

        self.n_iter += 1

    def is_flush_iteration(self) -> bool:
        """
        Return whether the current iteration is a forced-decoding iteration.
        """

        return (
            (self.n_iter + 1) % self.n_new_line == 0
        )

    def should_flush(self, force: bool = False) -> bool:
        """
        Whether the current transcript should be flushed.

        Normal flush:
            n_iter % n_new_line == 0

        Forced flush:
            force == True
        """

        return (
            force
            or self.n_iter % self.n_new_line == 0
        )

    def on_flush(self) -> None:
        """
        Reset utterance-related state after a transcript flush.

        The global stream timeline and iteration counter are preserved.

        The process-observation position is synchronized to the current
        stream position so that audio already observed before the flush
        is not counted again.
        """

        self.active_speech = False
        self.silence_samples = 0

        self.prev_process_stream_samples = 0
        self.prev_inference_start_timing = 0.0

    # ----------------------------------------------------------------------
    # Timeline
    # ----------------------------------------------------------------------

    @property
    def stream_ms(self) -> int:
        """
        Current streaming position in milliseconds.
        """

        return int(
            self.stream_samples * 1000 / self.sample_rate
        )
