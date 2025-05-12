import ctypes
from ctypes import POINTER, Structure, c_bool, c_size_t,  c_int32, c_float, c_char_p, c_void_p


class GreedyParams(Structure):
    _fields_ = [
        ("best_of", c_int32),
    ]


class BeamSearchParams(Structure):
    _fields_ = [
        ("beam_size", c_int32),
        ("patience", c_float),
    ]


class WhisperFullParams(Structure):
    _fields_ = [
        ("strategy", c_int32),
        #
        ("n_threads", c_int32),
        ("n_max_text_ctx", c_int32),
        ("offset_ms", c_int32),
        ("duration_ms", c_int32),
        #
        ("translate", c_bool),
        ("no_context", c_bool),
        ("no_timestamps", c_bool),
        ("single_segment", c_bool),
        ("print_special", c_bool),
        ("print_progress", c_bool),
        ("print_realtime", c_bool),
        ("print_timestamps", c_bool),
        #
        ("token_timestamps", c_bool),
        ("thold_pt", c_float),
        ("thold_ptsum", c_float),
        ("max_len", c_int32),
        ("split_on_word", c_bool),
        ("max_tokens", c_int32),
        #
        ("debug_mode", c_bool),
        ("audio_ctx", c_int32),
        #
        ("tdrz_enable", c_bool),
        #
        ('suppress_regex', c_char_p),
        #
        ("initial_prompt", c_char_p),
        ("prompt_tokens", POINTER(c_int32)),
        ("prompt_n_tokens", c_int32),
        #
        ("language", c_char_p),
        ("detect_language", c_bool),
        #
        ("suppress_blank", c_bool),
        ("suppress_nst", c_bool),
        #
        ("temperature", c_float),
        ("max_initial_ts", c_float),
        ("length_penalty", c_float),
        #
        ("temperature_inc", c_float),
        ("entropy_thold", c_float),
        ("logprob_thold", c_float),
        ("no_speech_thold", c_float),
        #
        ("greedy", GreedyParams),
        ("beam_search", BeamSearchParams),
        #
        ("new_segment_callback", c_void_p),
        ("new_segment_callback_user_data", c_void_p),
        #
        ("progress_callback", c_void_p),
        ("progress_callback_user_data", c_void_p),
        #
        ("encoder_begin_callback", c_void_p),
        ("encoder_begin_callback_user_data", c_void_p),
        #
        ("abort_callback", c_void_p),
        ("abort_callback_user_data", c_void_p),
        #
        ("logits_filter_callback", c_void_p),
        ("logits_filter_callback_user_data", c_void_p),
        #
        ("grammar_rules", c_void_p),
        ("n_grammar_rules", c_size_t),
        ("i_start_rule", c_size_t),
        ("grammar_penalty", c_float),
    ]


class WhisperAheadsParams(ctypes.Structure):
    _fields_ = [
        ("n_heads", ctypes.c_size_t),
        ("heads", ctypes.c_void_p),
    ]


class WhisperContextParams(ctypes.Structure):
    _fields_ = [
        ("use_gpu", c_bool),
        ("flash_attn", c_bool),
        ("gpu_device", c_int32),  # CUDA device
        # [EXPERIMENTAL] Token-level timestamps with DTW
        ("dtw_token_timestamps", c_bool),
        ("dtw_aheads_preset", c_int32),
        #
        ("dtw_n_top", c_int32),
        ("dtw_aheads", WhisperAheadsParams),
        #
        ("dtw_mem_size", c_size_t),
    ]
