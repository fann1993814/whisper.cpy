import soundfile as sf

from whispercpy import WhisperCPP
from whispercpy.utils import to_timestamp

WHISPER_CPP_PATH = "../../whisper.cpp"

# file path
audio_wav = f"{WHISPER_CPP_PATH}/samples/jfk.wav"
asr_model_path = f"{WHISPER_CPP_PATH}/models/ggml-tiny.bin"
vad_model_path = f"{WHISPER_CPP_PATH}/models/ggml-silero-v5.1.2.bin"
library_path = f"{WHISPER_CPP_PATH}/build/src/libwhisper.dylib"

# reading audio
data, sr = sf.read(audio_wav, dtype='float32')

# load model
model = WhisperCPP(library_path, asr_model_path,
                   vad_model_path, use_gpu=False, verbose=True)

print('--------- Lib Version ---------')
print(model.get_version())


print('--------- VAD Result ----------')

# get vad results
for segment in model.vad(data):
    print(f'[{to_timestamp(segment.t0, False)}' +
          " --> " + f'{to_timestamp(segment.t1, False)}]')

print('--------- ASR Result ----------')

# get asr results
for segment in model.transcribe(data, language='en', beam_size=5, token_timestamps=True):
    print(f'[{to_timestamp(segment.t0, False)}' +
          " --> " + f'{to_timestamp(segment.t1, False)}] ' + segment.text)
    print('--------- Token Info ----------')
    print('\n'.join([f'[{to_timestamp(token.t0, False)}' +
          " --> " + f'{to_timestamp(token.t1, False)}] {token.text}' for token in segment.tokens]))
    print('-------------------------------')
