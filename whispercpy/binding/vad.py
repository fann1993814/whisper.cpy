import ctypes
from ctypes import (
    c_int32,
    c_float,
    c_char_p,
    c_void_p,
)

from .structs import (
    VADParams,
    VADContextParams,
)


class VADLibrary:
    """Low-level ctypes binding for whisper.cpp VAD APIs."""

    def __init__(self, lib_path: str):
        self.lib = ctypes.cdll.LoadLibrary(lib_path)

        self._setup_vad_functions()

    def _setup_vad_functions(self) -> None:
        lib = self.lib

        # === Context ===

        lib.whisper_vad_default_context_params.argtypes = []
        lib.whisper_vad_default_context_params.restype = VADContextParams

        lib.whisper_vad_init_from_file_with_params.argtypes = [
            c_char_p,
            VADContextParams,
        ]
        lib.whisper_vad_init_from_file_with_params.restype = c_void_p

        lib.whisper_vad_free.argtypes = [
            c_void_p,
        ]
        lib.whisper_vad_free.restype = c_void_p

        # === Params ===

        lib.whisper_vad_default_params.argtypes = []
        lib.whisper_vad_default_params.restype = VADParams

        # === Segments ===

        lib.whisper_vad_segments_from_samples.argtypes = [
            c_void_p,
            VADParams,
            ctypes.POINTER(c_float),
            c_int32,
        ]
        lib.whisper_vad_segments_from_samples.restype = c_void_p

        lib.whisper_vad_segments_n_segments.argtypes = [
            c_void_p,
        ]
        lib.whisper_vad_segments_n_segments.restype = c_int32

        lib.whisper_vad_segments_get_segment_t0.argtypes = [
            c_void_p,
            c_int32,
        ]
        lib.whisper_vad_segments_get_segment_t0.restype = c_float

        lib.whisper_vad_segments_get_segment_t1.argtypes = [
            c_void_p,
            c_int32,
        ]
        lib.whisper_vad_segments_get_segment_t1.restype = c_float

        lib.whisper_vad_free_segments.argtypes = [
            c_void_p,
        ]
        lib.whisper_vad_free_segments.restype = c_void_p
