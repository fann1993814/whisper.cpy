from __future__ import annotations

import ctypes
import threading

from ctypes import c_void_p
from threading import Thread
from typing import List, Optional

import numpy as np

from ..base.asr import ASRBase
from ..binding.structs import WhisperFullParams
from ..common.constant import (
    STREAMING_ENDING,
    WHISPER_SAMPLE_RATE,
)
from ..common.interface import (
    TranscriptSegment,
    TranscriptToken,
)
from .buffer import SlidingAudioBuffer
from .policy import BasePolicy, FixedStepPolicy
from .vad import WebRTCVAD


class StreamingASR(ASRBase):
    """
    Sliding-window streaming ASR based on whisper.cpp.

    The streaming session keeps a persistent whisper.cpp state and
    incrementally processes incoming audio using a sliding window.

    Responsibilities are divided into:

        SlidingAudioBuffer
            Audio buffering and sliding-window construction.

        BasePolicy
            Speech detection, decode scheduling, timing, and flushing.

        StreamingASR
            Streaming orchestration, whisper.cpp inference, and
            transcript post-processing.
    """

    def __init__(
        self,
        model_path: str,
        language: str,
        step_ms: int = 500,
        keep_ms: int = 250,
        length_ms: int = 30000,
        return_token: bool = False,
        use_gpu: bool = True,
        verbose: bool = True,
        policy: Optional[BasePolicy] = None,
        speech_detector: Optional[WebRTCVAD] = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            use_gpu=use_gpu,
            verbose=verbose,
        )

        if step_ms <= 0:
            raise ValueError(
                "step_ms must be positive"
            )

        if keep_ms < 0:
            raise ValueError(
                "keep_ms must be non-negative"
            )

        if length_ms <= 0:
            raise ValueError(
                "length_ms must be positive"
            )

        self.language = language

        # Preserve the original WhisperStream behavior:
        #
        #     keep_ms = min(keep_ms, step_ms)
        #     length_ms = max(length_ms, step_ms)
        #
        self.step_ms = step_ms
        self.keep_ms = min(keep_ms, step_ms)
        self.length_ms = max(length_ms, step_ms)

        self.sample_rate = WHISPER_SAMPLE_RATE

        self.n_samples_step = int(
            self.step_ms
            * 1e-3
            * self.sample_rate
        )

        self.n_samples_keep = int(
            self.keep_ms
            * 1e-3
            * self.sample_rate
        )

        self.n_samples_len = int(
            self.length_ms
            * 1e-3
            * self.sample_rate
        )

        self.return_token = return_token

        # ------------------------------------------------------------------
        # Audio buffer
        # ------------------------------------------------------------------

        self.buffer = SlidingAudioBuffer(
            step_samples=self.n_samples_step,
            keep_samples=self.n_samples_keep,
            length_samples=self.n_samples_len,
        )

        # ------------------------------------------------------------------
        # Streaming policy
        # ------------------------------------------------------------------

        if policy is None:
            self.policy = FixedStepPolicy(
                step_samples=self.n_samples_step,
                length_samples=self.n_samples_len,
                sample_rate=self.sample_rate,
                speech_detector=speech_detector,
            )
        else:
            self.policy = policy

        # ------------------------------------------------------------------
        # whisper.cpp session state
        # ------------------------------------------------------------------

        self.state: Optional[c_void_p] = None

        self.params: Optional[WhisperFullParams] = None

        # Amount of audio overlap used by the previous flushed window.
        #
        # This is used by post_process() to map Whisper timestamps
        # back to the streaming timeline.
        self.prev_inference_overlap_ms = 0

        # ------------------------------------------------------------------
        # Transcript state
        # ------------------------------------------------------------------

        self.transcript_list: List[
            TranscriptSegment
        ] = []

        self.transcript = TranscriptSegment(
            index=0
        )

        # feed() remains asynchronous, but actual streaming state
        # modification is serialized.
        self._thread_lock = threading.Lock()

    # ----------------------------------------------------------------------
    # Session lifecycle
    # ----------------------------------------------------------------------

    def start(
        self,
        params: Optional[WhisperFullParams] = None,
    ) -> None:
        """
        Start a streaming session.

        A whisper.cpp state is created once and reused across feed()
        calls.
        """

        if self.state is not None:
            raise RuntimeError(
                "Streaming session is already running"
            )

        state = self.init_state()

        try:
            if params is None:
                params = self.init_params(
                    strategy=0,
                    best_of=1,
                    translate=False,
                    no_timestamps=not self.return_token,
                    no_context=True,
                    single_segment=True,
                    print_progress=False,
                    print_special=False,
                    print_realtime=False,
                    print_timestamps=False,
                    token_timestamps=self.return_token,
                    language=self.language
                )

                owns_params = True

            else:
                owns_params = False

        except Exception:
            self.free_state(state)
            raise

        self.state = state
        self.params = params

        # Reset streaming state.
        self.buffer.clear()
        self.policy.reset()

        self.prev_inference_overlap_ms = 0

        self.transcript_list = []

        self.transcript = TranscriptSegment(
            index=0
        )

    def stop(self) -> None:
        """
        Stop the streaming session and release session resources.
        """

        if self.state is not None:
            self.free_state(self.state)
            self.state = None

        if self.params is not None:
            self.free_params(self.params)
            self.params = None

        self.buffer.clear()
        self.policy.reset()

    def reset(self) -> None:
        """
        Reset the current streaming session.
        """

        self.stop()

        self.transcript_list = []

        self.transcript = TranscriptSegment(
            index=0
        )

        self.prev_inference_overlap_ms = 0

    def close(self) -> None:
        """
        Close the streaming ASR and release base resources.
        """

        self.stop()
        super().close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ----------------------------------------------------------------------
    # Audio input
    # ----------------------------------------------------------------------

    def feed(
        self,
        audio: np.ndarray,
    ) -> Thread:
        """
        Feed float32 PCM samples into the streaming ASR.

        Processing is asynchronous.

        Multiple feed() calls may create multiple worker threads, but
        actual streaming state updates are serialized by _thread_lock.
        """

        audio = np.ascontiguousarray(
            audio,
            dtype=np.float32,
        )

        if audio.size == 0:
            return Thread()

        thread = Thread(
            target=self._process_audio,
            args=(audio,),
        )

        thread.start()

        return thread

    def _process_audio(
        self,
        audio: np.ndarray,
    ) -> None:
        """
        Process one incoming audio chunk.

        The processing order intentionally follows the original
        WhisperStream.pipe() implementation.
        """

        with self._thread_lock:

            # --------------------------------------------------------------
            # 1. Update streaming timeline.
            #
            # Original:
            #
            #     self.stream_ms += ...
            #
            # This happens as soon as audio arrives, even if the audio
            # has not accumulated enough samples for processing yet.
            # --------------------------------------------------------------

            self.policy.add_samples(
                len(audio)
            )

            # --------------------------------------------------------------
            # 2. Append incoming audio to the pending buffer.
            #
            # Original:
            #
            #     self.pcmf32_new += chunk
            # --------------------------------------------------------------

            self.buffer.append(audio)

            # --------------------------------------------------------------
            # 3. Check whether enough NEW audio has accumulated.
            #
            # Buffer owns this decision.
            #
            # This replaces the old:
            #
            #     self.policy.ready(...)
            # --------------------------------------------------------------

            if not self.buffer.ready():
                return

            # --------------------------------------------------------------
            # 4. Build the sliding inference window.
            #
            # This corresponds to:
            #
            #     pcmf32_old[-n_samples_take:] + pcmf32_new
            #
            # in the original implementation.
            # --------------------------------------------------------------

            window = self.buffer.build_window()

            # --------------------------------------------------------------
            # 5. Check whether the previous overlap was fully preserved.
            #
            # If the previous window was truncated because the new
            # audio was already too large, the old timestamp overlap
            # can no longer be trusted.
            #
            # Original:
            #
            #     if len(pcmf32_old) > n_samples_take:
            #         prev_inference_overlap_ms = 0
            # --------------------------------------------------------------

            if not self.buffer.overlap_preserved:
                self.prev_inference_overlap_ms = 0

            # --------------------------------------------------------------
            # 6. Speech detection / processing policy.
            #
            # Important:
            # VAD receives the COMPLETE sliding window, not only the
            # newly received audio.
            # --------------------------------------------------------------

            if not self.policy.should_process(
                window
            ):
                return

            is_eou = self.policy.is_utterance_end()

            # --------------------------------------------------------------
            # 7. Decide whether Whisper inference should actually run.
            #
            # The policy may skip inference when the previous inference
            # is still slower than incoming audio.
            # --------------------------------------------------------------

            if self.policy.should_decode() or is_eou:

                # Record inference start BEFORE calling Whisper.
                self.policy.on_decode_start()

                self.transcribe(
                    window
                )

            # --------------------------------------------------------------
            # 8. Mark one processable streaming iteration.
            #
            # This happens even if should_decode() returned False.
            #
            # This preserves:
            #
            #     self.n_iter += 1
            #
            # from the original implementation.
            # --------------------------------------------------------------

            self.policy.on_iteration()

            # --------------------------------------------------------------
            # 9. Periodically flush the transcript.
            # --------------------------------------------------------------

            if is_eou:
                self.flush(force=True)
            else:
                self.flush()

            return

    def pipe(
        self,
        chunk: bytes,
    ) -> Thread:
        """
        Feed raw float32 PCM bytes asynchronously.
        """

        if chunk == STREAMING_ENDING:
            return self.end()

        if not chunk:
            return Thread()

        audio = np.frombuffer(
            chunk,
            dtype=np.float32,
        )

        return self.feed(audio)

    def end(self) -> Thread:
        """
        Finish the current stream and force a transcript flush.
        """

        def worker() -> None:
            with self._thread_lock:
                self.flush(force=True)

        thread = Thread(
            target=worker
        )

        thread.start()

        return thread

    # ----------------------------------------------------------------------
    # Whisper inference
    # ----------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
    ) -> None:
        """
        Run Whisper inference on the current sliding window.

        The inference input is padded to at least 2 seconds, matching
        the original WhisperStream implementation.
        """

        if self.state is None or self.params is None:
            raise RuntimeError(
                "Streaming session has not been started"
            )

        audio = np.ascontiguousarray(
            audio,
            dtype=np.float32,
        )

        input_samples = len(audio)

        segments = self.inference(
            audio,
            self.state,
            self.params,
        )

        self.post_process(
            segments,
            input_samples=input_samples,
        )

    # ----------------------------------------------------------------------
    # Transcript processing
    # ----------------------------------------------------------------------

    def post_process(
        self,
        segments: List[TranscriptSegment],
        input_samples: int,
    ) -> None:
        """
        Convert Whisper timestamps to streaming timestamps.

        Whisper timestamps are relative to the current inference window.
        They are rescaled and shifted into the global streaming timeline.
        """

        if input_samples <= 0:
            return

        transcribe_ms = int(
            input_samples
            * 1000
            / self.sample_rate
        )

        if transcribe_ms <= 0:
            return

        # Preserve the original timestamp scaling behavior.
        rescale = (
            transcribe_ms
            - self.prev_inference_overlap_ms
        ) / transcribe_ms

        stream_ms = self.policy.stream_ms

        # TranscriptSegment timestamps are represented in 10 ms units.
        self.transcript.t1 = (
            stream_ms // 10
        )

        self.transcript.t0 = (
            stream_ms
            + self.prev_inference_overlap_ms
            - transcribe_ms
        ) // 10

        text = ""
        tokens: List[
            TranscriptToken
        ] = []

        for segment in segments:

            text += segment.text

            if not self.return_token:
                continue

            for token in segment.tokens:

                if token.t0 is None or token.t1 is None:
                    continue

                t0 = int(
                    self.transcript.t0
                    + token.t0 * rescale
                )

                t1 = int(
                    self.transcript.t0
                    + token.t1 * rescale
                )

                # Token starts after the current streaming endpoint.
                if t0 > self.transcript.t1:
                    continue

                # Clamp token end to the current streaming endpoint.
                if t1 > self.transcript.t1:
                    t1 = self.transcript.t1

                tokens.append(
                    TranscriptToken(
                        token.text,
                        t0,
                        t1,
                    )
                )

        self.transcript.text = text
        self.transcript.tokens = tokens

    # ----------------------------------------------------------------------
    # Transcript flushing
    # ----------------------------------------------------------------------

    def flush(
        self,
        force: bool = False,
    ) -> None:
        """
        Flush the current transcript.

        The last keep_ms of the current inference window is retained
        for the next window to mitigate word-boundary issues.

        Flush is a transaction across:

            Buffer
                keep audio tail

            Transcript
                commit current transcript and create a new one

            Policy
                reset speech / inference timing state
        """

        if not self.policy.should_flush(
            force
        ):
            return

        # --------------------------------------------------------------
        # 1. Keep the audio tail for the next inference window.
        #
        # Original:
        #
        #     self.pcmf32_old = self.pcmf32[-n_samples_keep:]
        # --------------------------------------------------------------

        self.buffer.keep_tail()

        # --------------------------------------------------------------
        # 2. Tell timestamp processing that the next inference window
        #    begins with keep_ms of overlap.
        # --------------------------------------------------------------

        self.prev_inference_overlap_ms = (
            self.keep_ms
        )

        # --------------------------------------------------------------
        # 3. Commit the current transcript if it contains text.
        # --------------------------------------------------------------

        if self.transcript.text:
            self.transcript_list.append(
                self.transcript
            )

        # --------------------------------------------------------------
        # 4. Create the next transcript segment.
        # --------------------------------------------------------------

        self.transcript = TranscriptSegment(
            index=len(self.transcript_list)
        )

        # --------------------------------------------------------------
        # 5. Reset policy-level speech / inference state.
        #
        # n_iter and stream_samples intentionally remain unchanged.
        # --------------------------------------------------------------

        self.policy.on_flush()

    # ----------------------------------------------------------------------
    # Results
    # ----------------------------------------------------------------------

    def get_transcripts(
        self,
    ) -> List[TranscriptSegment]:
        """
        Return all committed transcript segments.
        """

        return self.transcript_list

    def get_transcript(
        self,
    ) -> TranscriptSegment:
        """
        Return the current, not-yet-committed transcript segment.
        """

        return self.transcript
