import os
import ctypes
import numpy as np

from typing import List, Optional
from ctypes import c_int32, c_int64, c_float, c_char_p, c_void_p

from .structs import WhisperContextParams, WhisperFullParams, WhisperTokenData
from .interface import TranscriptSegment, TranscriptToken


class WhisperCPP:
    def __init__(
            self,
            lib_path: str,
            model_path: str,
            use_gpu: bool = True):

        # === Init lib ===
        self.lib = None

        # === Check path ===
        assert os.path.exists(lib_path) == True
        assert os.path.exists(model_path) == True

        # === Load lib ===
        self.lib = ctypes.cdll.LoadLibrary(lib_path)

        # === Function prototypes ===
        self.lib.whisper_context_default_params.argtypes = []
        self.lib.whisper_context_default_params.restype = WhisperContextParams
        self.lib.whisper_init_from_file_with_params.argtypes = [
            c_char_p, WhisperContextParams]
        self.lib.whisper_init_from_file_with_params.restype = c_void_p
        self.lib.whisper_full_default_params.argtypes = [c_int32]
        self.lib.whisper_full_default_params.restype = WhisperFullParams
        self.lib.whisper_init_state.argtypes = [c_void_p]
        self.lib.whisper_init_state.restype = c_void_p
        self.lib.whisper_full_with_state.argtypes = [
            c_void_p, c_void_p, WhisperFullParams, ctypes.POINTER(c_float), c_int32]
        self.lib.whisper_full_with_state.restype = c_int32
        self.lib.whisper_full_n_segments_from_state.argtypes = [
            c_void_p]
        self.lib.whisper_full_n_segments_from_state.restype = c_int32
        self.lib.whisper_full_get_segment_text_from_state.argtypes = [
            c_void_p, c_int32]
        self.lib.whisper_full_get_segment_text_from_state.restype = c_char_p
        self.lib.whisper_full_get_segment_t0_from_state.argtypes = [
            c_void_p, c_int32]
        self.lib.whisper_full_get_segment_t0_from_state.restype = c_int64
        self.lib.whisper_full_get_segment_t1_from_state.argtypes = [
            c_void_p, c_int32]
        self.lib.whisper_full_get_segment_t1_from_state.restype = c_int64
        self.lib.whisper_full_n_tokens_from_state.argtypes = [
            c_void_p, c_int32]
        self.lib.whisper_full_n_tokens_from_state.restype = c_int32
        self.lib.whisper_full_get_token_data_from_state.argtypes = [
            c_void_p, c_int32, c_int32]
        self.lib.whisper_full_get_token_data_from_state.restype = WhisperTokenData
        self.lib.whisper_free.argtypes = [c_void_p]
        self.lib.whisper_free.restype = c_void_p
        self.lib.whisper_token_to_str.argtypes = [c_void_p, c_int32]
        self.lib.whisper_token_to_str.restype = c_char_p
        self.lib.whisper_free_state.argtypes = [c_void_p]
        self.lib.whisper_free_state.restype = c_void_p

        # === Load whisper context param ===
        cparam = self.lib.whisper_context_default_params()
        cparam.use_gpu = use_gpu
        cparam.flash_attn |= use_gpu

        # === Load whisper model ===
        self.ctx = self.lib.whisper_init_from_file_with_params(
            model_path.encode(), cparam)

        assert self.ctx is not None

    def __del__(self) -> None:
        if self.lib:
            self.lib.whisper_free(self.ctx)

    def init_state(self,) -> c_void_p:
        return self.lib.whisper_init_state(self.ctx)

    def free_state(self, state: c_void_p) -> None:
        return self.lib.whisper_free_state(state)

    def init_params(
            self,
            strategy: int = 0,  # 0 is greedy (default), 1 is beam mode
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
            prompt_tokens: List[int] = [],
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

        # follow default whisper.cpp setting
        params = self.lib.whisper_full_default_params(strategy)

        params.n_threads = n_threads if n_threads is not None else params.n_threads
        params.n_max_text_ctx = n_max_text_ctx if n_max_text_ctx is not None else params.n_max_text_ctx
        params.offset_ms = offset_ms if offset_ms is not None else params.offset_ms
        params.duration_ms = duration_ms if duration_ms is not None else params.duration_ms

        params.translate = translate if translate is not None else params.translate
        params.no_context = no_context if no_context is not None else params.no_context
        params.no_timestamps = no_timestamps if no_timestamps is not None else params.no_timestamps
        params.single_segment = single_segment if single_segment is not None else params.single_segment
        params.print_special = print_special if print_special is not None else params.print_special
        params.print_progress = print_progress if print_progress is not None else params.print_progress
        params.print_realtime = print_realtime if print_realtime is not None else params.print_realtime
        params.print_timestamps = print_timestamps if print_timestamps is not None else params.print_timestamps

        params.token_timestamps = token_timestamps if token_timestamps is not None else params.token_timestamps
        params.thold_pt = thold_pt if thold_pt is not None else params.thold_pt
        params.thold_ptsum = thold_ptsum if thold_ptsum is not None else params.thold_ptsum
        params.max_len = max_len if max_len is not None else params.max_len
        params.split_on_word = split_on_word if split_on_word is not None else params.split_on_word
        params.max_tokens = max_tokens if max_tokens is not None else params.max_tokens

        params.suppress_regex = suppress_regex.encode(
            'utf-8') if suppress_regex else params.suppress_regex

        params.initial_prompt = initial_prompt.encode(
            'utf-8') if initial_prompt else params.initial_prompt
        params.prompt_tokens = np.array(
            prompt_tokens, dtype=np.int32).ctypes.data_as(ctypes.POINTER(c_int32)) if prompt_tokens else params.prompt_tokens
        params.prompt_n_tokens = len(
            prompt_tokens) if prompt_tokens else params.prompt_n_tokens

        params.language = language.encode(
            'utf-8') if language else params.language
        params.detect_language = detect_language if detect_language is not None else params.detect_language

        params.suppress_blank = suppress_blank if suppress_blank is not None else params.suppress_blank
        params.suppress_nst = suppress_nst if suppress_nst is not None else params.suppress_nst

        params.temperature = temperature if temperature is not None else params.temperature
        params.max_initial_ts = max_initial_ts if max_initial_ts is not None else params.max_initial_ts
        params.length_penalty = length_penalty if length_penalty is not None else params.length_penalty

        params.temperature_inc = temperature_inc if temperature_inc is not None else params.temperature_inc
        params.entropy_thold = entropy_thold if entropy_thold is not None else params.entropy_thold
        params.logprob_thold = logprob_thold if logprob_thold is not None else params.logprob_thold
        params.no_speech_thold = no_speech_thold if no_speech_thold is not None else params.no_speech_thold

        params.greedy.best_of = best_of if best_of is not None else params.greedy.best_of
        params.beam_search.beam_size = beam_size if beam_size is not None else params.beam_search.beam_size

        return params

    def inferece(
            self,
            data: np.ndarray,
            state: Optional[c_void_p] = None,
            params: Optional[WhisperFullParams] = None) -> List[TranscriptSegment]:

        free = True if not state else False
        state = self.init_state() if not state else state
        params = self.init_params() if not params else params

        ret = self.lib.whisper_full_with_state(self.ctx, state, params, data.ctypes.data_as(
            ctypes.POINTER(c_float)), len(data))

        assert ret == 0

        segments = []

        for i in range(self.lib.whisper_full_n_segments_from_state(state)):
            text = self.lib.whisper_full_get_segment_text_from_state(
                state, i).decode("utf-8")

            t0 = self.lib.whisper_full_get_segment_t0_from_state(
                state, i) if not params.no_timestamps else None
            t1 = self.lib.whisper_full_get_segment_t1_from_state(
                state, i) if not params.no_timestamps else None

            tokens = []
            n_tokens = self.lib.whisper_full_n_tokens_from_state(state, i)

            for j in range(n_tokens):
                token_data = self.lib.whisper_full_get_token_data_from_state(
                    state, i, j)
                token_text = self.lib.whisper_token_to_str(
                    self.ctx, token_data.id).decode('utf-8')

                if params.token_timestamps:
                    tokens.append(TranscriptToken(
                        token_text, token_data.t0, token_data.t1))
                else:
                    tokens.append(TranscriptToken(token_text))

            segments.append(TranscriptSegment(i, text, tokens, t0, t1))

        if free:
            self.free_state(state)

        return segments

    def transcribe(
            self,
            audio: np.ndarray,
            language: str,
            beam_size: int = 5,
            translate: bool = False,
            token_timestamps: bool = False,
            thold_pt: float = 0.01,
            thold_ptsum: float = 0.01,
            max_len: int = 0,
            split_on_word: bool = False,
            max_tokens: int = 0,
            suppress_blank: bool = True,
            suppress_nst: bool = False,
            temperature: float = 0.0,
            max_initial_ts: float = 1.0,
            length_penalty: float = -1.0,
            temperature_inc: float = 0.2,
            entropy_thold: float = 2.4,
            logprob_thold: float = -1.0,
            no_speech_thold: float = 0.6) -> List[TranscriptSegment]:

        # use beam search
        params = self.init_params(
            strategy=1,
            translate=translate,
            print_special=False,
            print_progress=False,
            print_realtime=False,
            print_timestamps=False,
            token_timestamps=token_timestamps,
            thold_pt=thold_pt,
            thold_ptsum=thold_ptsum,
            max_len=max_len,
            split_on_word=split_on_word,
            max_tokens=max_tokens,
            language=language,
            suppress_blank=suppress_blank,
            suppress_nst=suppress_nst,
            temperature=temperature,
            max_initial_ts=max_initial_ts,
            length_penalty=length_penalty,
            temperature_inc=temperature_inc,
            entropy_thold=entropy_thold,
            logprob_thold=logprob_thold,
            no_speech_thold=no_speech_thold,
            beam_size=beam_size,)

        return self.inferece(audio, params=params)
