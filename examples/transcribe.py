import soundfile as sf

from whispercpy import WhisperASR, SileroVAD
from whispercpy.common import to_timestamp

WHISPER_CPP_PATH = "../../whisper.cpp"

asr = WhisperASR(
    lib_path=f"{WHISPER_CPP_PATH}/build/bin/libwhisper.dylib",
    model_path=f"{WHISPER_CPP_PATH}/models/ggml-tiny.bin",
    use_gpu=True
)

vad = SileroVAD(
    lib_path=f"{WHISPER_CPP_PATH}/build/bin/libwhisper.dylib",
    model_path=f"{WHISPER_CPP_PATH}/models/ggml-silero-v6.2.0.bin",
)

print('------- Library Version -------')
print(asr.get_version())

audio, sr = sf.read(f"{WHISPER_CPP_PATH}/samples/jfk.wav", dtype='float32')

print('--------- VAD Result ----------')

# get vad results
for segment in vad.detect(audio):
    print(f'[{to_timestamp(segment.t0, False)}' +
          " --> " + f'{to_timestamp(segment.t1, False)}]')

print('--------- ASR Result ----------')

# get asr results
for segment in asr.transcribe(audio, language='en', beam_size=5, token_timestamps=True):
    print(f'[{to_timestamp(segment.t0, False)}' +
          " --> " + f'{to_timestamp(segment.t1, False)}] ' + segment.text)
    print('--------- Token Info ----------')
    print('\n'.join([f'[{to_timestamp(token.t0, False)}' +
          " --> " + f'{to_timestamp(token.t1, False)}] {token.text}' for token in segment.tokens]))
    print('-------------------------------')
