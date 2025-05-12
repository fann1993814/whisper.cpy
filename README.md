# whispercpy

Python wrapper for [whisper.cpp](https://github.com/ggml-org/whisper.cpp/)

# Highlight

1. Lightweight, using `ctypes.CDLL` to call functions from the `libwhisper` shared library.

2. Migrate the `whisper-stream` functions to deal with live streaming case for async-processing

# Index
<!-- TOC -->
* [Preparing](#preparing)
* [Usage](#usage)
  * [Basic Audio Transcribe](#basic-audio-transcribe)
  * [Live Streaming](#live-streaming)
* [License](#license)
<!-- TOC -->

# Preparing

## 1. Prepare `whisper.cpp` library

Clone whisper.cpp, then get into it

```sh
# clone whisper.cpp
git clone https://github.com/ggml-org/whisper.cpp.git

# nevagation into this folder
cd whisper.cpp/

# checkout the stable version (current supporting)
git checkout v1.7.5
```

Build library

```sh
# build  whisper.cpp
cmake -B build
cmake --build build --config Release
```

Download ggml models

```sh
sh ./models/download-ggml-model.sh [base|small|large]
```

## 2. Install `whispercpy`

Install from source:

```sh
pip install git+https://github.com/fann1993814/whispercpy.git
```

# Usage

## Basic Audio Transcribe
Below these steps, you can trace [trancribe.py](./examples/trancribe.py)

### 1. Share library, model, and testing audio setting

```py
# WHISPER_CPP_PATH is the whisper.cpp project location

audio_wav = f"{WHISPER_CPP_PATH}/samples/jfk.wav"
model_path = f"{WHISPER_CPP_PATH}/models/ggml-tiny.bin"
library_path = f"{WHISPER_CPP_PATH}/build/src/libwhisper.dylib" # Mac: dylib, Linux: so, Win: dll
```

### 2. Read testing audio of whisper.cpp

```py
import soundfile as sf

data, sr = sf.read(audio_wav, dtype='float32')
```
### 3. Load library and model with whispercpy, and transcribe, and get transcript results

```py
from whispercpy import WhipserCPP
from whispercpy.utils import to_timestamp


model = WhipserCPP(library_path, model_path, use_gpu=True)

transcripts = model.transcribe(data, language='en', beam_size=5)

for segment in transcripts:
    print(f'[{to_timestamp(segment.t0, False)}' +
          " --> " + f'{to_timestamp(segment.t1, False)}] ' + segment.text)

# Result
# [00:00:00.000 --> 00:00:10.400]  And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```
- `to_timestamp` can translate the time unit from whisper.cpp into a formal repesenation

# Live Streaming
Below these steps, you can trace [live.py](./examples/live.py)

**Note: for realtime inference,**
  - `tiny/base/small` for cpu
  - `medium/large/large-v2/large-v3` for gpu

### 1. Load core engine and steaming decoder with library and model

```py
from whispercpy import WhipserCPP, WhisperStream

core = WhipserCPP(lib_path, model_path, use_gpu=False)
asr = WhisperStream(core, language='en')
```

### 2. Callback function setting
```py
count = 0

def callback(indata, frames, time, status):
    global count

    chunk = indata.copy().tobytes()
    asr.pipe(chunk)
    transcript = asr.get_transcript()
    transcripts = asr.get_transcripts()

    if len(transcripts) > count:
        print("\r"+transcripts[-1].text)
        print('--')
        count += 1
    else:
        print(f"\r{transcript.text}", end="", flush=True)

```
- `asr.pipe`: a threading function for async to process audio for transcribing continuously
- `asr.get_transcript`: get the current transcirption
- `asr.get_transcripts`: get whole transcirptions

### 3. Setting microphone recording

```py
import sounddevice as sd
from whispercpy.constant import STREAMING_ENDING

samplerate = 16000
block_duration = 1.0
block_size = int(samplerate * block_duration)
channels = 1

# Recording
try:
    with sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        callback=callback, blocksize=block_size, dtype='float32'):
        print("🎤 Recording for ASR... Press Ctrl+C to stop.")
        while True:
            sd.sleep(1000)
except KeyboardInterrupt:
    print("⏹️ Recording stopped.")
    # send end signal
    asr.pipe(STREAMING_ENDING)

# Result
#
# Testing. Testing. Can you hear me? This is a asr testing.
# --
# I'm trying to speak English to testing our ASR system. Can you give me a response?
# --
# ^C⏹️ Recording stopped.
# [00:00:00.000 --> 00:00:10.000]  Testing. Testing. Can you hear me? This is a sour testing.
# [00:00:13.000 --> 00:00:23.000]  I'm trying to speak English to testing our ASR system. Can you give me a response?
```
- `STREAMING_ENDING`: a singal for stopping transcribing

# License
This project follow [whisper.cpp](https://github.com/ggml-org/whisper.cpp/) license as MIT