import os
import ctypes

import numpy as np

from typing import Dict, List, Optional
from ctypes import c_int32, c_float, c_void_p

from ..binding import WhisperLibrary
from ..binding.structs import (
    GGML_LOG_CALLBACK,
    WhisperFullParams,
)
from ..common.utils import empty_log_callback
from ..common.interface import TranscriptSegment, TranscriptToken


class ASRBase:
    """
    Base class for Whisper-based ASR implementations.

    This class manages the shared Whisper runtime:
        - Whisper library
        - Whisper model context
        - Whisper state creation/release
        - Whisper inference parameters
        - Whisper version
        - resource cleanup

    Offline and streaming ASR implementations should inherit
    from this class.
    """

    def __init__(
            self,
            model_path: str,
            lib_path: str,
            use_gpu: bool = True,
            verbose: bool = True):

        # === Check paths ===
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Whisper ASR model not found: {model_path}"
            )

        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"Whisper library not found: {lib_path}"
            )

        self.model_path = model_path
        self.vad_model_path = None
        self.use_gpu = use_gpu
        self.verbose = verbose

        # === Load library ===
        self.whisper = WhisperLibrary(lib_path)
        self.lib = self.whisper.lib

        # === Whisper context ===
        self.ctx: Optional[c_void_p] = None

        # Keep ctypes callback alive.
        self._log_callback = None

        # Keep Python objects referenced by WhisperFullParams alive.
        #
        # WhisperFullParams stores raw pointers for fields such as:
        #   - prompt_tokens
        #   - suppress_regex
        #   - initial_prompt
        #   - language
        #
        # The C library does not own these Python objects, so we need
        # to keep them alive while the corresponding params are in use.
        self._param_buffers: Dict[int, object] = {}

        # === Initialize model ===
        self._init_model()

    def _add_vad(self, vad_model_path: str) -> None:
        """Add a VAD model to the Whisper context."""

        if not os.path.exists(vad_model_path):
            raise FileNotFoundError(
                f"Whisper VAD model not found: {vad_model_path}"
            )
        else:
            self.vad_model_path = vad_model_path

    def _init_model(self) -> None:
        """Initialize the Whisper model context."""

        context_params = self.lib.whisper_context_default_params()
        context_params.use_gpu = self.use_gpu

        self.ctx = (
            self.lib.whisper_init_from_file_with_params_no_state(
                self.model_path.encode("utf-8"),
                context_params,
            )
        )

        if not self.ctx:
            raise RuntimeError(
                f"Failed to initialize Whisper model: "
                f"{self.asr_model_path}"
            )

        # === Disable whisper.cpp log output ===
        if not self.verbose:
            self._log_callback = GGML_LOG_CALLBACK(
                empty_log_callback
            )

            self.lib.whisper_log_set(
                self._log_callback,
                None,
            )

    # ------------------------------------------------------------------
    # Whisper state
    # ------------------------------------------------------------------

    def init_state(self) -> c_void_p:
        """
        Create a new Whisper state.

        Offline ASR can create a temporary state for each inference.
        Streaming ASR can keep a state alive across multiple chunks.
        """

        if not self.ctx:
            raise RuntimeError(
                "Whisper model is not initialized"
            )

        state = self.lib.whisper_init_state(self.ctx)

        if not state:
            raise RuntimeError(
                "Failed to initialize Whisper state"
            )

        return state

    def free_state(self, state: c_void_p) -> None:
        """Release a Whisper state."""

        if state:
            self.lib.whisper_free_state(state)

    # ------------------------------------------------------------------
    # Whisper parameters
    # ------------------------------------------------------------------

    def init_params(
            self,
            strategy: int = 0,
            #
            n_threads: Optional[int] = None,
            n_max_text_ctx: Optional[int] = None,
            offset_ms: Optional[int] = None,
            duration_ms: Optional[int] = None,
            #
            translate: Optional[bool] = None,
            no_context: Optional[bool] = None,
            no_timestamps: Optional[bool] = None,
            single_segment: Optional[bool] = None,
            print_special: Optional[bool] = None,
            print_progress: Optional[bool] = None,
            print_realtime: Optional[bool] = None,
            print_timestamps: Optional[bool] = None,
            #
            token_timestamps: Optional[bool] = None,
            thold_pt: Optional[float] = None,
            thold_ptsum: Optional[float] = None,
            max_len: Optional[int] = None,
            split_on_word: Optional[bool] = None,
            max_tokens: Optional[int] = None,
            #
            suppress_regex: Optional[str] = None,
            #
            initial_prompt: Optional[str] = None,
            carry_initial_prompt: Optional[bool] = None,
            prompt_tokens: Optional[List[int]] = None,
            #
            language: Optional[str] = None,
            detect_language: Optional[bool] = None,
            #
            suppress_blank: Optional[bool] = None,
            suppress_nst: Optional[bool] = None,
            #
            temperature: Optional[float] = None,
            max_initial_ts: Optional[float] = None,
            length_penalty: Optional[float] = None,
            #
            temperature_inc: Optional[float] = None,
            entropy_thold: Optional[float] = None,
            logprob_thold: Optional[float] = None,
            no_speech_thold: Optional[float] = None,
            #
            best_of: Optional[int] = None,
            beam_size: Optional[int] = None,
    ) -> WhisperFullParams:
        """
        Create Whisper full-inference parameters.

        Parameters that are not explicitly specified retain the
        whisper.cpp default values.

        Args:
            strategy:
                Whisper sampling strategy.
                0 = greedy, 1 = beam search.

            prompt_tokens:
                Optional list of token IDs used as the initial prompt.

        Returns:
            WhisperFullParams configured for Whisper inference.
        """

        if not self.ctx:
            raise RuntimeError(
                "Whisper model is not initialized"
            )

        # Follow whisper.cpp default settings.
        params = self.lib.whisper_full_default_params(strategy)

        # === General parameters ===

        params.n_threads = (
            n_threads
            if n_threads is not None
            else params.n_threads
        )

        params.n_max_text_ctx = (
            n_max_text_ctx
            if n_max_text_ctx is not None
            else params.n_max_text_ctx
        )

        params.offset_ms = (
            offset_ms
            if offset_ms is not None
            else params.offset_ms
        )

        params.duration_ms = (
            duration_ms
            if duration_ms is not None
            else params.duration_ms
        )

        # === Decoding / behavior ===

        params.translate = (
            translate
            if translate is not None
            else params.translate
        )

        params.no_context = (
            no_context
            if no_context is not None
            else params.no_context
        )

        params.no_timestamps = (
            no_timestamps
            if no_timestamps is not None
            else params.no_timestamps
        )

        params.single_segment = (
            single_segment
            if single_segment is not None
            else params.single_segment
        )

        params.print_special = (
            print_special
            if print_special is not None
            else params.print_special
        )

        params.print_progress = (
            print_progress
            if print_progress is not None
            else params.print_progress
        )

        params.print_realtime = (
            print_realtime
            if print_realtime is not None
            else params.print_realtime
        )

        params.print_timestamps = (
            print_timestamps
            if print_timestamps is not None
            else params.print_timestamps
        )

        # === Token / timestamp parameters ===

        params.token_timestamps = (
            token_timestamps
            if token_timestamps is not None
            else params.token_timestamps
        )

        params.thold_pt = (
            thold_pt
            if thold_pt is not None
            else params.thold_pt
        )

        params.thold_ptsum = (
            thold_ptsum
            if thold_ptsum is not None
            else params.thold_ptsum
        )

        params.max_len = (
            max_len
            if max_len is not None
            else params.max_len
        )

        params.split_on_word = (
            split_on_word
            if split_on_word is not None
            else params.split_on_word
        )

        params.max_tokens = (
            max_tokens
            if max_tokens is not None
            else params.max_tokens
        )

        # === String parameters ===

        if suppress_regex is not None:
            suppress_regex_bytes = suppress_regex.encode("utf-8")
            params.suppress_regex = suppress_regex_bytes
        else:
            suppress_regex_bytes = None

        if initial_prompt is not None:
            initial_prompt_bytes = initial_prompt.encode("utf-8")
            params.initial_prompt = initial_prompt_bytes
        else:
            initial_prompt_bytes = None

        if carry_initial_prompt is not None:
            params.carry_initial_prompt = carry_initial_prompt

        # === Prompt tokens ===

        if prompt_tokens:
            prompt_array = np.asarray(
                prompt_tokens,
                dtype=np.int32,
            )

            params.prompt_tokens = (
                prompt_array.ctypes.data_as(
                    ctypes.POINTER(c_int32)
                )
            )

            params.prompt_n_tokens = len(prompt_tokens)

        else:
            prompt_array = None

        # === Language ===

        if language is not None:
            language_bytes = language.encode("utf-8")
            params.language = language_bytes
        else:
            language_bytes = None

        params.detect_language = (
            detect_language
            if detect_language is not None
            else params.detect_language
        )

        # === Suppression ===

        params.suppress_blank = (
            suppress_blank
            if suppress_blank is not None
            else params.suppress_blank
        )

        params.suppress_nst = (
            suppress_nst
            if suppress_nst is not None
            else params.suppress_nst
        )

        # === Sampling ===

        params.temperature = (
            temperature
            if temperature is not None
            else params.temperature
        )

        params.max_initial_ts = (
            max_initial_ts
            if max_initial_ts is not None
            else params.max_initial_ts
        )

        params.length_penalty = (
            length_penalty
            if length_penalty is not None
            else params.length_penalty
        )

        params.temperature_inc = (
            temperature_inc
            if temperature_inc is not None
            else params.temperature_inc
        )

        params.entropy_thold = (
            entropy_thold
            if entropy_thold is not None
            else params.entropy_thold
        )

        params.logprob_thold = (
            logprob_thold
            if logprob_thold is not None
            else params.logprob_thold
        )

        params.no_speech_thold = (
            no_speech_thold
            if no_speech_thold is not None
            else params.no_speech_thold
        )

        # === Decoding strategy ===

        params.greedy.best_of = (
            best_of
            if best_of is not None
            else params.greedy.best_of
        )

        params.beam_search.beam_size = (
            beam_size
            if beam_size is not None
            else params.beam_search.beam_size
        )

        # --------------------------------------------------------------
        # Keep Python objects alive.
        #
        # WhisperFullParams contains raw pointers to these objects.
        # The params object itself does not own the underlying memory.
        # --------------------------------------------------------------

        self._param_buffers[id(params)] = (
            prompt_array,
            suppress_regex_bytes,
            initial_prompt_bytes,
            language_bytes,
        )

        # === VAD ===

        if self.vad_model_path:
            params.vad_model_path = self.vad_model_path.encode("utf-8")

        return params

    def free_params(self, params: WhisperFullParams) -> None:
        """
        Release Python-side buffers associated with Whisper parameters.

        This should be called after the C API is guaranteed to no longer
        use the corresponding WhisperFullParams.
        """

        self._param_buffers.pop(id(params), None)

    # ------------------------------------------------------------------
    # Whisper Inference and Processing Segments
    # ------------------------------------------------------------------

    def inference(
        self,
        audio: np.ndarray,
        state: c_void_p,
        params: WhisperFullParams,
    ) -> List[TranscriptSegment]:
        """
        Run whisper.cpp inference.
        """

        audio = np.ascontiguousarray(
            audio,
            dtype=np.float32,
        )

        ret = self.lib.whisper_full_with_state(
            self.ctx,
            state,
            params,
            audio.ctypes.data_as(
                ctypes.POINTER(c_float)
            ),
            len(audio),
        )

        if ret != 0:
            raise RuntimeError(
                "whisper_full_with_state() failed "
                f"with error code {ret}"
            )

        return self.get_segments(
            state,
            params,
        )

    def get_segments(
        self,
        state: c_void_p,
        params: WhisperFullParams,
    ) -> List[TranscriptSegment]:
        """
        Extract decoded segments and tokens from whisper.cpp state.
        """

        segments: List[
            TranscriptSegment
        ] = []

        n_segments = (
            self.lib.whisper_full_n_segments_from_state(
                state
            )
        )

        for i in range(n_segments):
            text = (
                self.lib.whisper_full_get_segment_text_from_state(
                    state,
                    i,
                )
                .decode(
                    "utf-8",
                    errors="replace",
                )
            )

            # Segment timestamps are only requested when timestamps
            # are enabled.
            if params.no_timestamps:
                t0 = None
                t1 = None
            else:
                t0 = (
                    self.lib.whisper_full_get_segment_t0_from_state(
                        state,
                        i,
                    )
                )

                t1 = (
                    self.lib.whisper_full_get_segment_t1_from_state(
                        state,
                        i,
                    )
                )

            tokens: List[
                TranscriptToken
            ] = []

            n_tokens = (
                self.lib.whisper_full_n_tokens_from_state(
                    state,
                    i,
                )
            )

            for j in range(n_tokens):

                token_data = (
                    self.lib.whisper_full_get_token_data_from_state(
                        state,
                        i,
                        j,
                    )
                )

                token_text = (
                    self.lib.whisper_token_to_str(
                        self.ctx,
                        token_data.id,
                    )
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )

                if params.token_timestamps:
                    tokens.append(
                        TranscriptToken(
                            token_text,
                            token_data.t0,
                            token_data.t1,
                        )
                    )
                else:
                    tokens.append(
                        TranscriptToken(
                            token_text
                        )
                    )

            segments.append(
                TranscriptSegment(
                    i,
                    text,
                    tokens,
                    t0,
                    t1,
                )
            )

        return segments

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        """Return the whisper.cpp version."""

        return self.lib.whisper_version().decode("utf-8")

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the Whisper model context."""

        if self.ctx:
            self.lib.whisper_free(self.ctx)
            self.ctx = None

        self._param_buffers.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
