import os
import ctypes

import numpy as np

from typing import List, Optional
from ctypes import c_float, c_void_p

from ..binding import VADLibrary
from ..binding.structs import VADParams
from ..common.interface import VoiceSegment
from ..common.constant import FLT_MAX


class SileroVAD:
    """
    This class manages the Silero VAD runtime:
        - VAD library
        - VAD model context
        - VAD inference parameters
        - VAD segment extraction
        - resource cleanup
    """

    def __init__(
            self,
            lib_path: str,
            model_path: str,
            verbose: bool = True):

        # === Check paths ===
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"VAD library not found: {lib_path}"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"VAD model not found: {model_path}"
            )

        self.model_path = model_path
        self.verbose = verbose

        # === Load library ===
        self.vad = VADLibrary(lib_path)
        self.lib = self.vad.lib

        # === VAD context ===
        self.vctx: Optional[c_void_p] = None

        # === Initialize model ===
        self._init_model()

    # ------------------------------------------------------------------
    # VAD model
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        """Initialize the VAD model context."""

        context_params = (
            self.lib.whisper_vad_default_context_params()
        )

        self.vctx = (
            self.lib.whisper_vad_init_from_file_with_params(
                self.model_path.encode("utf-8"),
                context_params,
            )
        )

        if not self.vctx:
            raise RuntimeError(
                f"Failed to initialize VAD model: "
                f"{self.model_path}"
            )

    # ------------------------------------------------------------------
    # VAD parameters
    # ------------------------------------------------------------------

    def init_params(
            self,
            threshold: float = 0.5,
            min_speech_duration_ms: int = 250,
            min_silence_duration_ms: int = 100,
            max_speech_duration_s: float = FLT_MAX,
            speech_pad_ms: int = 30,
            samples_overlap: float = 0.1,
    ) -> VADParams:
        """
        Create VAD parameters.

        Parameters that are not explicitly specified use
        whisper.cpp default values.
        """

        if not self.vctx:
            raise RuntimeError(
                "VAD model is not initialized"
            )

        params = self.lib.whisper_vad_default_params()

        params.threshold = threshold
        params.min_speech_duration_ms = (
            min_speech_duration_ms
        )
        params.min_silence_duration_ms = (
            min_silence_duration_ms
        )
        params.max_speech_duration_s = (
            max_speech_duration_s
        )
        params.speech_pad_ms = speech_pad_ms
        params.samples_overlap = samples_overlap

        return params

    # ------------------------------------------------------------------
    # VAD inference
    # ------------------------------------------------------------------

    def detect(
            self,
            audio: np.ndarray,
            threshold: float = 0.5,
            min_speech_duration_ms: int = 250,
            min_silence_duration_ms: int = 100,
            max_speech_duration_s: float = FLT_MAX,
            speech_pad_ms: int = 30,
            samples_overlap: float = 0.1,
    ) -> List[VoiceSegment]:
        """
        Detect speech segments from audio samples.

        Args:
            audio:
                Mono float32 audio samples.

            threshold:
                VAD speech probability threshold.

            min_speech_duration_ms:
                Minimum speech duration.

            min_silence_duration_ms:
                Minimum silence duration.

            max_speech_duration_s:
                Maximum speech segment duration.

            speech_pad_ms:
                Padding added around speech segments.

            samples_overlap:
                Overlap between VAD processing windows.

        Returns:
            A list of VoiceSegment objects.
        """

        if not self.vctx:
            raise RuntimeError(
                "VAD model is not initialized"
            )

        # --------------------------------------------------------------
        # Prepare audio
        # --------------------------------------------------------------

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.ndim != 1:
            raise ValueError(
                "VAD expects a mono audio array."
            )

        if audio.size == 0:
            return []

        audio = np.ascontiguousarray(audio)

        # --------------------------------------------------------------
        # Create parameters
        # --------------------------------------------------------------

        params = self.init_params(
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            max_speech_duration_s=max_speech_duration_s,
            speech_pad_ms=speech_pad_ms,
            samples_overlap=samples_overlap,
        )

        # --------------------------------------------------------------
        # Run VAD
        # --------------------------------------------------------------

        segments = (
            self.lib.whisper_vad_segments_from_samples(
                self.vctx,
                params,
                audio.ctypes.data_as(
                    ctypes.POINTER(c_float)
                ),
                len(audio),
            )
        )

        if not segments:
            raise RuntimeError(
                "whisper_vad_segments_from_samples failed"
            )

        try:
            n_segments = (
                self.lib.whisper_vad_segments_n_segments(
                    segments
                )
            )

            results: List[VoiceSegment] = []

            for i in range(n_segments):
                t0 = (
                    self.lib
                    .whisper_vad_segments_get_segment_t0(
                        segments,
                        i,
                    )
                )

                t1 = (
                    self.lib
                    .whisper_vad_segments_get_segment_t1(
                        segments,
                        i,
                    )
                )

                results.append(
                    VoiceSegment(
                        index=i,
                        t0=float(t0),
                        t1=float(t1),
                    )
                )

            return results

        finally:
            self.lib.whisper_vad_free_segments(
                segments
            )

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the VAD model context."""

        if self.vctx:
            self.lib.whisper_vad_free(self.vctx)
            self.vctx = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
