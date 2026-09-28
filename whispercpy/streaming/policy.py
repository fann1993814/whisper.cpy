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

    - whether the current audio window should be processed
    - whether Whisper decoding should run
    - when the transcript should be flushed
    - streaming / inference timing state

    The policy does not own audio buffering.

    Audio availability is handled by the audio buffer.
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
    def should_process(self, audio: np.ndarray) -> bool:
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

    This policy preserves the scheduling behavior of the original
    WhisperStream implementation:

    - speech detection
    - continuous processing while speech is active
    - inference-time protection
    - periodic forced decoding
    - periodic transcript flushing

    The policy does not own audio data.
    """

    def __init__(
        self,
        step_samples: int,
        length_samples: int,
        sample_rate: int = 16000,
        speech_detector: Optional[SpeechDetector] = None,
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

        self.step_samples = step_samples
        self.length_samples = length_samples
        self.sample_rate = sample_rate

        self.speech_detector = speech_detector

        # ------------------------------------------------------------------
        # Speech state
        # ------------------------------------------------------------------

        # Corresponds to the original:
        #
        #     self.active_speech = False
        #
        # Once speech has been detected, subsequent windows are processed
        # until flush() resets this state.
        self.active_speech = False

        # ------------------------------------------------------------------
        # Iteration state
        # ------------------------------------------------------------------

        # Corresponds to:
        #
        #     self.n_iter = 0
        #
        # This is incremented after a processable streaming iteration,
        # regardless of whether Whisper inference actually runs.
        self.n_iter = 0

        # Corresponds to:
        #
        #     self.n_new_line = max(
        #         1,
        #         int(length_ms / step_ms - 1)
        #     )
        #
        # Since length_samples / step_samples has the same ratio as
        # length_ms / step_ms, we can calculate it directly in samples.
        self.n_new_line = max(
            1,
            int(length_samples / step_samples - 1),
        )

        # ------------------------------------------------------------------
        # Streaming timeline
        # ------------------------------------------------------------------

        # Total number of audio samples received.
        #
        # This corresponds to the original stream_ms counter.
        self.stream_samples = 0

        # ------------------------------------------------------------------
        # Inference timing
        # ------------------------------------------------------------------

        # Start position of the audio consumed by the previous inference.
        self.prev_inference_start_stream_samples = 0

        # Wall-clock time when the previous inference started.
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

        self.n_iter = 0

        self.stream_samples = 0

        self.prev_inference_start_stream_samples = 0
        self.prev_inference_start_timing = 0.0

    # ----------------------------------------------------------------------
    # Streaming timeline
    # ----------------------------------------------------------------------

    def add_samples(self, n_samples: int) -> None:
        """
        Add newly received samples to the streaming timeline.

        Note that this counts ALL received audio, even audio that has
        not yet reached the processing step.

        This matches the original WhisperStream behavior where:

            self.stream_ms += len(chunk) * ...

        happened immediately when a chunk arrived.
        """

        if n_samples < 0:
            raise ValueError(
                "n_samples must be non-negative"
            )

        self.stream_samples += n_samples

    # ----------------------------------------------------------------------
    # Speech / processing policy
    # ----------------------------------------------------------------------

    def should_process(self, audio: np.ndarray) -> bool:
        """
        Decide whether the current window should be processed.

        Behavior matches the original WhisperStream:

        1. If speech is already active, process the window.
        2. If no speech detector is configured, treat the window as speech.
        3. Otherwise, start processing only when speech is detected.
        """

        # Once speech has started, continue processing subsequent
        # windows until the transcript is flushed.
        if self.active_speech:
            return True

        # Without a speech detector, every window is considered speech.
        if self.speech_detector is None:
            self.active_speech = True
            return True

        # Start a new speech segment only when speech is detected.
        if self.speech_detector(audio):
            self.active_speech = True
            return True

        return False

    # ----------------------------------------------------------------------
    # Decode scheduling
    # ----------------------------------------------------------------------

    def should_decode(self) -> bool:
        """
        Decide whether Whisper inference should run.

        This preserves the original WhisperStream logic:

            if prev_inference_spend_time <=
                    prev_inference_consume_audio_time
               or (n_iter + 1) % n_new_line == 0:

                transcribe()

        In other words:

        - run immediately if there has not been a previous inference
        - otherwise allow decoding when inference is keeping up
        - periodically force decoding to avoid starving the decoder
        """

        # No previous inference.
        #
        # Original:
        #
        #     prev_inference_start_timing > 0
        #
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

        That matches the original:

            self.n_iter += 1

        which happened after the speech-processing block regardless of
        whether transcribe() was actually called.
        """

        self.n_iter += 1

    def is_flush_iteration(self) -> bool:
        """
        Return whether the current iteration is a forced-decoding iteration.

        The original implementation checked:

            (self.n_iter + 1) % self.n_new_line == 0

        before incrementing n_iter.

        Therefore this method intentionally uses n_iter + 1.
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
        Reset speech/inference state after a transcript flush.

        The iteration counter and stream position are intentionally
        preserved.
        """

        self.active_speech = False

        # Match the original flush():

        #     self.prev_inference_start_timing = 0

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
