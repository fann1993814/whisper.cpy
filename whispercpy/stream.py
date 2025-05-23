import time
import numpy as np

from typing import List

import transcribe

from .core import WhipserCPP
from .utils import run_aysnc, is_speech
from .constant import C_FLOAT_TO_BYTES_RATIO, WHISPER_SAMPLE_RATE, STREAMING_ENDING
from .interface import TranscriptSegment, TranscriptToken


class WhisperStream:
    def __init__(
            self,
            core: WhipserCPP,
            language: str,
            step_ms: int = 500,
            keep_ms: int = 200,
            length_ms: int = 10000,
            token_mode: bool = False):
        self.core = core

        self.step_ms = step_ms
        self.keep_ms = min(keep_ms, step_ms)
        self.length_ms = max(length_ms, step_ms)

        self.n_samples_step = int(
            (1e-3*self.step_ms)*WHISPER_SAMPLE_RATE*C_FLOAT_TO_BYTES_RATIO)
        self.n_samples_len = int((1e-3*self.length_ms)
                                 * WHISPER_SAMPLE_RATE*C_FLOAT_TO_BYTES_RATIO)
        self.n_samples_keep = int(
            (1e-3*self.keep_ms)*WHISPER_SAMPLE_RATE*C_FLOAT_TO_BYTES_RATIO)
        self.n_samples_2s = int(
            (1e-3*2000.0)*WHISPER_SAMPLE_RATE*C_FLOAT_TO_BYTES_RATIO)

        self.active_speech = False
        self.prev_inference_start_steam_ms = 0
        self.prev_inference_start_timing = -1
        self.prev_inference_overlap_ms = 0

        self.pcmf32 = b''
        self.pcmf32_old = b''
        self.pcmf32_new = b''

        self.n_iter = 0
        self.n_new_line = max(1, int(self.length_ms /
                                     self.step_ms - 1))
        self.stream_ms = 0
        self.transcript_list = []
        self.transcript = TranscriptSegment(index=len(self.transcript_list))

        self.state = self.core.init_state()

        self.token_mode = token_mode

        # for streaming setting, with greedy decode (fast)
        self.params = self.core.init_params(
            strategy=0,
            best_of=1,
            translate=False,
            no_timestamps=False if token_mode else True,
            no_context=True,
            single_segment=True,
            print_progress=False,
            print_special=False,
            print_realtime=False,
            print_timestamps=False,
            token_timestamps=True if token_mode else False,
            language=language)

    def __del__(self,):
        if self.state:
            self.core.lib.whisper_free_state(self.state)

    @run_aysnc
    def pipe(self, chunk: bytes):
        if chunk != STREAMING_ENDING:
            # counting
            self.stream_ms += len(
                chunk) * 1000 / (WHISPER_SAMPLE_RATE * C_FLOAT_TO_BYTES_RATIO)

            # collect
            self.pcmf32_new += chunk

            # enough to process
            if len(self.pcmf32_new) >= self.n_samples_step:
                n_samples_new = len(self.pcmf32_new)

                # take up to params.length_ms audio from previous iteration
                n_samples_take = min(len(self.pcmf32_old), max(
                    0, self.n_samples_keep + self.n_samples_len - n_samples_new))

                # sliding window move, so do not minus overlap
                if len(self.pcmf32_old) > n_samples_take:
                    self.prev_inference_overlap_ms = 0

                self.pcmf32 = self.pcmf32_old[-n_samples_take:]
                self.pcmf32 += self.pcmf32_new

                self.pcmf32_old = self.pcmf32

                # speech trigger
                if self.active_speech or is_speech(np.frombuffer(self.pcmf32, dtype=np.float32)):
                    # get previous inference time spend
                    prev_inference_spend_time = int((
                        time.time() - self.prev_inference_start_timing) * 100) / 100 if self.prev_inference_start_timing > 0 else 0
                    # get previous audio time consume
                    prev_inference_consume_audio_time = (
                        self.stream_ms - self.prev_inference_start_steam_ms) / 1000

                    # avoid inference slow, skip frequent inference
                    if prev_inference_spend_time <= prev_inference_consume_audio_time or (self.n_iter + 1) % self.n_new_line == 0:
                        # run the inference
                        self.prev_inference_start_timing = time.time()
                        self.prev_inference_start_steam_ms = self.stream_ms
                        self.transcribe()

                    self.n_iter += 1
                    self.active_speech = True

                    # flush
                    self.flush()

                # reset pcm32f_new
                self.pcmf32_new = b''
        else:
            self.flush(True)

    def transcribe(self,):
        data = self.pcmf32 + b''.join([b'\0' for _ in range(self.n_samples_2s - len(
            self.pcmf32))]) if len(self.pcmf32) < self.n_samples_2s else self.pcmf32
        data = np.frombuffer(data, dtype=np.float32)
        segments = self.core.inferece(data, self.state, self.params)
        self.post_process(segments)

    def post_process(self, segments: List[TranscriptSegment]):
        transcribe_ms = int(len(self.pcmf32) * 1000 /
                            ((WHISPER_SAMPLE_RATE * C_FLOAT_TO_BYTES_RATIO)))
        rescale = (transcribe_ms -
                   self.prev_inference_overlap_ms) / transcribe_ms

        self.transcript.t1 = self.stream_ms // 10
        self.transcript.t0 = (
            self.stream_ms + self.prev_inference_overlap_ms - transcribe_ms) // 10

        text = ''
        tokens = []

        for segment in segments:
            text += segment.text
            if self.token_mode:
                for token in segment.tokens:
                    t0 = int(self.transcript.t0 + token.t0 * rescale)
                    t1 = int(self.transcript.t0 + token.t1 * rescale)

                    if t0 > self.transcript.t1:
                        continue
                    if t1 > self.transcript.t1:
                        t1 = self.transcript.t1

                    tokens.append(TranscriptToken(token.text, t0, t1))

        self.transcript.text = text
        self.transcript.tokens = tokens

    def flush(self, force: bool = False):
        if self.n_iter % self.n_new_line == 0 or force:
            # keep part of the audio for next iteration to try to mitigate word boundary issues
            self.pcmf32_old = self.pcmf32[-self.n_samples_keep:]
            self.prev_inference_overlap_ms = self.keep_ms

            # Append transcript into transcript_list
            if self.transcript.text:
                self.transcript_list.append(self.transcript)

            # Reset current transcript
            self.transcript = TranscriptSegment(
                index=len(self.transcript_list))

            # Reset speech
            self.active_speech = False

            # Reset
            self.prev_inference_start_timing = 0

    def get_transcripts(self,) -> List[TranscriptSegment]:
        return self.transcript_list

    def get_transcript(self,) -> TranscriptSegment:
        return self.transcript
