import ctypes
from ctypes import (
    c_int32,
    c_int64,
    c_float,
    c_char_p,
    c_void_p,
)

from .structs import (
    GGML_LOG_CALLBACK,
    WhisperContextParams,
    WhisperFullParams,
    WhisperTokenData,
)


class WhisperLibrary:
    """Low-level ctypes binding for whisper.cpp ASR APIs."""

    def __init__(self, lib_path: str):
        self.lib = ctypes.cdll.LoadLibrary(lib_path)

        self._setup_asr_functions()
        self._setup_general_functions()

    def _setup_asr_functions(self) -> None:
        lib = self.lib

        # === Context ===

        lib.whisper_context_default_params.argtypes = []
        lib.whisper_context_default_params.restype = WhisperContextParams

        lib.whisper_init_from_file_with_params_no_state.argtypes = [
            c_char_p,
            WhisperContextParams,
        ]
        lib.whisper_init_from_file_with_params_no_state.restype = c_void_p

        lib.whisper_free.argtypes = [c_void_p]
        lib.whisper_free.restype = c_void_p

        # === State ===

        lib.whisper_init_state.argtypes = [c_void_p]
        lib.whisper_init_state.restype = c_void_p

        lib.whisper_free_state.argtypes = [c_void_p]
        lib.whisper_free_state.restype = c_void_p

        # === Full params ===

        lib.whisper_full_default_params.argtypes = [c_int32]
        lib.whisper_full_default_params.restype = WhisperFullParams

        # === Inference ===

        lib.whisper_full_with_state.argtypes = [
            c_void_p,
            c_void_p,
            WhisperFullParams,
            ctypes.POINTER(c_float),
            c_int32,
        ]
        lib.whisper_full_with_state.restype = c_int32

        # === Segments ===

        lib.whisper_full_n_segments_from_state.argtypes = [
            c_void_p,
        ]
        lib.whisper_full_n_segments_from_state.restype = c_int32

        lib.whisper_full_get_segment_text_from_state.argtypes = [
            c_void_p,
            c_int32,
        ]
        lib.whisper_full_get_segment_text_from_state.restype = c_char_p

        lib.whisper_full_get_segment_t0_from_state.argtypes = [
            c_void_p,
            c_int32,
        ]
        lib.whisper_full_get_segment_t0_from_state.restype = c_int64

        lib.whisper_full_get_segment_t1_from_state.argtypes = [
            c_void_p,
            c_int32,
        ]
        lib.whisper_full_get_segment_t1_from_state.restype = c_int64

        # === Tokens ===

        lib.whisper_full_n_tokens_from_state.argtypes = [
            c_void_p,
            c_int32,
        ]
        lib.whisper_full_n_tokens_from_state.restype = c_int32

        lib.whisper_full_get_token_data_from_state.argtypes = [
            c_void_p,
            c_int32,
            c_int32,
        ]
        lib.whisper_full_get_token_data_from_state.restype = WhisperTokenData

        lib.whisper_token_to_str.argtypes = [
            c_void_p,
            c_int32,
        ]
        lib.whisper_token_to_str.restype = c_char_p

        lib.whisper_full_get_segment_no_speech_prob_from_state.argtypes = [
            c_void_p,
            c_int32,
        ]
        lib.whisper_full_get_segment_no_speech_prob_from_state.restype = c_float

    def _setup_general_functions(self) -> None:
        lib = self.lib

        # === Version ===

        lib.whisper_version.argtypes = []
        lib.whisper_version.restype = c_char_p

        # === Logging ===

        lib.whisper_log_set.argtypes = [
            GGML_LOG_CALLBACK,
            c_void_p,
        ]
        lib.whisper_log_set.restype = c_void_p
