import soundfile as sf

from whispercpy import WhipserCPP
from whispercpy.utils import to_timestamp

WHISPER_CPP_PATH = "../../whisper.cpp"

# file path
audio_wav = f"{WHISPER_CPP_PATH}/samples/jfk.wav"
model_path = f"{WHISPER_CPP_PATH}/models/ggml-tiny.bin"
library_path = f"{WHISPER_CPP_PATH}/build/src/libwhisper.dylib"

# reading audio
data, sr = sf.read(audio_wav, dtype='float32')

# load model
model = WhipserCPP(library_path, model_path, use_gpu=True)

# run transcirbe
res = model.transcribe(data, language='en', beam_size=5)

# get results
for segment in res:
    print(f'[{to_timestamp(segment.t0, False)}' +
          " --> " + f'{to_timestamp(segment.t1, False)}] ' + segment.text)
