# whisper.cpy

Python wrapper for [whisper.cpp](https://github.com/ggml-org/whisper.cpp/)

# Highlight

1. Lightweight, using `ctypes.CDLL` to call functions from the `libwhisper` shared library.

2. Migrate the [`whisper-stream`](https://github.com/ggml-org/whisper.cpp/tree/master/examples/stream) functions to deal with live streaming case for async-processing

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

Clone whisper.cpp, then build it

```sh
# clone whisper.cpp
git clone https://github.com/ggml-org/whisper.cpp

# nevagation into this folder
cd whisper.cpp/

# checkout the stable version (current supporting)
git checkout v1.7.5

# build whisper.cpp
cmake -B build
cmake --build build --config Release
```

Download ggml models

```sh
sh ./models/download-ggml-model.sh [tiny|base|small|large]
```

## 2. Install `whisper.cpy`

Install from source:

```sh
pip install git+https://github.com/fann1993814/whisper.cpy
```

# Usage

## Basic Audio Transcribe
Follow below steps, and trace [trancribe.py](./examples/trancribe.py)

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

### 3. Load library and model with whisper.cpy, and transcribe, and get transcript results

```py
from whispercpy import WhipserCPP
from whispercpy.utils import to_timestamp


model = WhipserCPP(library_path, model_path, use_gpu=True)

transcripts = model.transcribe(data, language='en', beam_size=5, token_timestamps=True)

for segment in transcripts:
    print(f'[{to_timestamp(segment.t0, False)}' +
          " --> " + f'{to_timestamp(segment.t1, False)}] ' + segment.text)
    print('-------------------------------')
    print('\n'.join([f'[{to_timestamp(token.t0, False)}' +
          " --> " + f'{to_timestamp(token.t1, False)}] {token.text}' for token in segment.tokens]))
    print('-------------------------------')

# Result
# [00:00:00.000 --> 00:00:10.400]  And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
# -------------------------------
# [00:00:00.000 --> 00:00:00.000] [_BEG_]
# [00:00:00.320 --> 00:00:00.320]  And
# [00:00:00.330 --> 00:00:00.530]  so
# [00:00:00.680 --> 00:00:00.740] ,
# [00:00:00.740 --> 00:00:00.950]  my
# [00:00:00.950 --> 00:00:01.590]  fellow
# [00:00:01.590 --> 00:00:02.100]  Americans
# [00:00:02.550 --> 00:00:03.000] ,
# [00:00:03.290 --> 00:00:03.650]  ask
# [00:00:04.010 --> 00:00:04.280]  not
# [00:00:04.650 --> 00:00:05.200]  what
# [00:00:05.410 --> 00:00:05.560]  your
# [00:00:05.650 --> 00:00:06.410]  country
# [00:00:06.410 --> 00:00:06.750]  can
# [00:00:06.750 --> 00:00:06.920]  do
# [00:00:07.010 --> 00:00:07.490]  for
# [00:00:07.490 --> 00:00:07.970]  you
# [00:00:08.170 --> 00:00:08.170] ,
# [00:00:08.190 --> 00:00:08.430]  ask
# [00:00:08.430 --> 00:00:08.750]  what
# [00:00:08.910 --> 00:00:09.040]  you
# [00:00:09.040 --> 00:00:09.350]  can
# [00:00:09.350 --> 00:00:09.500]  do
# [00:00:09.500 --> 00:00:09.710]  for
# [00:00:09.720 --> 00:00:09.980]  your
# [00:00:09.990 --> 00:00:10.350]  country
# [00:00:10.470 --> 00:00:10.500] .
# [00:00:10.500 --> 00:00:10.500] [_TT_525]
# -------------------------------
```
- `to_timestamp` can translate the time unit from whisper.cpp into a formal repesenation

# Live Streaming
Follow below steps, and trace [live.py](./examples/live.py)

**Note: for realtime inference,**
  - `tiny/base/small` for cpu
  - `medium/large/large-v2/large-v3` for gpu

### 1. Load core engine and steaming decoder with library and model

```py
from whispercpy import WhipserCPP, WhisperStream

core = WhipserCPP(lib_path, model_path, use_gpu=False)
asr = WhisperStream(core, language='en', return_token=True)
```

### 2. Callback setting
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

### 3. Microphone recording setting

```py
import sounddevice as sd
from whispercpy.constant import STREAMING_ENDING

samplerate = 16000
block_duration = 0.25
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
    asr.pipe(STREAMING_ENDING).join()

# Result
#
# 🎤 Recording for ASR... Press Ctrl+C to stop.
# This is my voice test.
# --
# Can you hear me?
# --
# ^C⏹️ Recording stopped.
# [00:00:00.800 --> 00:00:11.000]  This is my voice test.
# -------------------------------
# [00:00:00.800 --> 00:00:00.800] [_BEG_]
# [00:00:00.800 --> 00:00:01.490]  This
# [00:00:01.850 --> 00:00:01.850]  is
# [00:00:01.890 --> 00:00:02.200]  my
# [00:00:02.200 --> 00:00:03.080]  voice
# [00:00:03.080 --> 00:00:03.710]  test
# [00:00:03.710 --> 00:00:08.290] .
# [00:00:08.300 --> 00:00:11.000] [_TT_150]
# -------------------------------
# [00:00:11.300 --> 00:00:21.500]  Can you hear me?
# -------------------------------
# [00:00:11.300 --> 00:00:11.300] [_BEG_]
# [00:00:11.300 --> 00:00:11.790]  Can
# [00:00:12.050 --> 00:00:12.280]  you
# [00:00:12.280 --> 00:00:12.940]  hear
# [00:00:12.940 --> 00:00:13.070]  me
# [00:00:13.070 --> 00:00:17.960] ?
# [00:00:17.960 --> 00:00:21.500] [_TT_100]
# -------------------------------
```

- `STREAMING_ENDING`: a singal for stopping transcribing, and use `join()` for waiting last thread complete.

# License
This project follows [whisper.cpp](https://github.com/ggml-org/whisper.cpp/) license as MIT