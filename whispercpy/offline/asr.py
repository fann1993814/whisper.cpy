import ctypes

import numpy as np

from typing import List, Optional
from ctypes import c_float, c_void_p

from ..base.asr import ASRBase
from ..binding.structs import WhisperFullParams
from ..common.interface import TranscriptSegment, TranscriptToken


class WhisperASR(ASRBase):
    """
    Offline Whisper ASR implementation.

    This class performs full-audio transcription using whisper.cpp.
    """

    def transcribe(
            self,
            audio: np.ndarray,
            language: str,
            beam_size: int = 5,
            translate: bool = False,
            #
            token_timestamps: bool = False,
            thold_pt: float = 0.01,
            thold_ptsum: float = 0.01,
            max_len: int = 0,
            split_on_word: bool = False,
            max_tokens: int = 0,
            #
            initial_prompt: str = '',
            carry_initial_prompt: Optional[bool] = None,
            #
            suppress_blank: bool = True,
            suppress_nst: bool = False,
            #
            temperature: float = 0.0,
            max_initial_ts: float = 1.0,
            length_penalty: float = -1.0,
            #
            temperature_inc: float = 0.2,
            entropy_thold: float = 2.4,
            logprob_thold: float = -1.0,
            no_speech_thold: float = 0.6,
    ) -> List[TranscriptSegment]:
        """
        Transcribe a complete audio signal using beam search.
        """

        # Initialize the Whisper state and parameters for transcription.
        state = self.init_state()
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

            initial_prompt=initial_prompt,
            carry_initial_prompt=carry_initial_prompt,

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

            beam_size=beam_size,
        )

        # Run inference with the provided audio, state, and parameters.
        results = self.inference(
            audio,
            state=state,
            params=params,
        )

        # Free the state and params if they were created in this method.
        self.free_state(state)
        self.free_params(params)

        return results
