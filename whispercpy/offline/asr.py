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

    def inference(
            self,
            data: np.ndarray,
            state: Optional[c_void_p] = None,
            params: Optional[WhisperFullParams] = None,
    ) -> List[TranscriptSegment]:
        """
        Run Whisper inference on a complete audio signal.

        Args:
            data:
                Audio samples as a NumPy array.

            state:
                Optional existing Whisper state.
                If not provided, a temporary state is created and
                released automatically after inference.

            params:
                Optional WhisperFullParams.
                If not provided, default parameters are created.

        Returns:
            A list of TranscriptSegment.
        """

        # Make sure the audio buffer is compatible with the C API.
        data = np.ascontiguousarray(
            data,
            dtype=np.float32,
        )

        # Track whether this method owns the state.
        owns_state = state is None

        if state is None:
            state = self.init_state()

        if params is None:
            params = self.init_params()

        try:
            ret = self.lib.whisper_full_with_state(
                self.ctx,
                state,
                params,
                data.ctypes.data_as(
                    ctypes.POINTER(c_float)
                ),
                len(data),
            )

            if ret != 0:
                raise RuntimeError(
                    f"whisper_full_with_state() failed "
                    f"with error code {ret}"
                )

            return self._get_segments(
                state,
                params,
            )

        finally:
            if owns_state:
                self.free_state(state)

                # params may contain Python-side buffers.
                self.free_params(params)

    def _get_segments(
            self,
            state: c_void_p,
            params: WhisperFullParams,
    ) -> List[TranscriptSegment]:
        """
        Extract transcription segments and tokens from a Whisper state.
        """

        segments = []

        n_segments = (
            self.lib.whisper_full_n_segments_from_state(
                state
            )
        )

        for i in range(n_segments):
            # ----------------------------------------------------------
            # Segment text
            # ----------------------------------------------------------

            text = (
                self.lib
                .whisper_full_get_segment_text_from_state(
                    state,
                    i,
                )
                .decode(
                    "utf-8",
                    errors="replace",
                )
            )

            # ----------------------------------------------------------
            # Segment timestamps
            # ----------------------------------------------------------

            if params.no_timestamps:
                t0 = None
                t1 = None
            else:
                t0 = (
                    self.lib
                    .whisper_full_get_segment_t0_from_state(
                        state,
                        i,
                    )
                )

                t1 = (
                    self.lib
                    .whisper_full_get_segment_t1_from_state(
                        state,
                        i,
                    )
                )

            # ----------------------------------------------------------
            # Tokens
            # ----------------------------------------------------------

            tokens = []

            n_tokens = (
                self.lib.whisper_full_n_tokens_from_state(
                    state,
                    i,
                )
            )

            for j in range(n_tokens):
                token_data = (
                    self.lib
                    .whisper_full_get_token_data_from_state(
                        state,
                        i,
                        j,
                    )
                )

                token_text = (
                    self.lib
                    .whisper_token_to_str(
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
                            token_text,
                        )
                    )

            # ----------------------------------------------------------
            # Segment
            # ----------------------------------------------------------

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

        return self.inference(
            audio,
            params=params,
        )
