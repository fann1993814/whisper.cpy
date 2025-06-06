import soundfile as sf

from whispercpy import WhisperCPP
from whispercpy.utils import to_timestamp

WHISPER_CPP_PATH = "../../whisper.cpp"

# file path
audio_wav = f"{WHISPER_CPP_PATH}/samples/jfk.wav"
model_path = f"{WHISPER_CPP_PATH}/models/ggml-tiny.bin"
library_path = f"{WHISPER_CPP_PATH}/build/src/libwhisper.dylib"

# reading audio
data, sr = sf.read(audio_wav, dtype='float32')

# load model
model = WhisperCPP(library_path, model_path, use_gpu=False)

# run transcirbe
res = model.transcribe(data, language='en', beam_size=5, token_timestamps=True)

# get results
for segment in res:
    print(f'[{to_timestamp(segment.t0, False)}' +
          " --> " + f'{to_timestamp(segment.t1, False)}] ' + segment.text)
    print('-------------------------------')
    print('\n'.join([f'[{to_timestamp(token.t0, False)}' +
          " --> " + f'{to_timestamp(token.t1, False)}] {token.text}' for token in segment.tokens]))
    print('-------------------------------')
