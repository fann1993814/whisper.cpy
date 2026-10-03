# whisper.cpy

Python wrapper for [whisper.cpp](https://github.com/ggml-org/whisper.cpp/)

# Highlight

1. Lightweight, using `ctypes.CDLL` to call functions from the `libwhisper` shared library.

2. Migrate the [`whisper-stream`](https://github.com/ggml-org/whisper.cpp/tree/master/examples/stream) functions to deal with live streaming case for async-processing

# Index
<!-- TOC -->
* [Preparing](#preparing)
* [Usage](#usage)
  * [Basic Audio Transcribe and VAD](#basic-audio-transcribe-and-voice-activity-detection)
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
git checkout v1.9.4

# build whisper.cpp
cmake -B build
cmake --build build --config Release
```

Download ggml models

```sh
# ASR model
sh ./models/download-ggml-model.sh [tiny|base|small|large]

# VAD model
sh ./models/download-vad-model.sh silero-v6.2.0
```

## 2. Install `whisper.cpy`

Install from source:

```sh
pip install git+https://github.com/fann1993814/whisper.cpy
```

# Usage

## Basic Audio Transcribe and Voice Activity Detection
Follow below steps, and trace [trancribe.py](./examples/trancribe.py)

### 1. Share library, model, and testing audio setting

```py
# WHISPER_CPP_PATH is the whisper.cpp project location

audio_wav = f"{WHISPER_CPP_PATH}/samples/jfk.wav"
asr_model_path = f"{WHISPER_CPP_PATH}/models/ggml-tiny.bin"
vad_model_path = f"{WHISPER_CPP_PATH}/models/ggml-silero-v6.2.0.bin"
lib_path = f"{WHISPER_CPP_PATH}/build/bin/libwhisper.dylib" # Mac: dylib, Linux: so, Win: dll
```

### 2. Read testing audio of whisper.cpp

```py
import soundfile as sf

data, sr = sf.read(audio_wav, dtype='float32')
```

### 3. Load library and model with whisper.cpy, and transcribe, and get transcript results

```py
import whispercpy

from whispercpy import WhisperASR, SileroVAD
from whispercpy.common import to_timestamp


# Library Initialization
whispercpy.set_lib_path(lib_path)


# Models Initialization

asr = WhisperASR(
    model_path=asr_model_path,
    lib_path=lib_path,
    use_gpu=True
)

vad = SileroVAD(
    model_path=vad_model_path,
    lib_path=lib_path,
)

# -------- VAD Detect ---------

for segment in vad.detect(data):
    print(f'[{to_timestamp(segment.t0, False)}' +
          " --> " + f'{to_timestamp(segment.t1, False)}]')

# -------- VAD Result ---------
# [00:00:00.320 --> 00:00:02.270]
# [00:00:03.270 --> 00:00:04.410]
# [00:00:05.380 --> 00:00:07.680]
# [00:00:08.160 --> 00:00:10.620]

# -------- ASR Transcribing --------

for segment in asr.transcribe(data, language='en', beam_size=5, token_timestamps=True):
    print(f'[{to_timestamp(segment.t0, False)}' +
          " --> " + f'{to_timestamp(segment.t1, False)}] ' + segment.text)
    print('--------- Token Info ----------')
    print('\n'.join([f'[{to_timestamp(token.t0, False)}' +
          " --> " + f'{to_timestamp(token.t1, False)}] {token.text}' for token in segment.tokens]))
    print('-------------------------------')

# -------- ASR Result --------
# [00:00:00.000 --> 00:00:10.400]  And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

# -------- Token Info --------
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
import whispercpy

from whispercpy import StreamingASR, WebRTCVAD
from whispercpy.common import to_timestamp

whispercpy.set_lib_path(lib_path)

asr = StreamingASR(
    model_path=model_path,
    language="en",
    step_ms=500,
    keep_ms=250,
    return_token=True,
    use_gpu=True,
    speech_detector=WebRTCVAD(),
)
```

### 2. Result printer setting
```py
import threading

stop_printer = threading.Event()


def result_loop():
    last_count = 0
    last_text = ""

    while not stop_printer.is_set():

        transcripts = asr.get_transcripts()
        transcript = asr.get_transcript()

        # New committed transcript
        if len(transcripts) > last_count:
            for item in transcripts[last_count:]:
                print(
                    f"\n[COMMITTED] {item.text}"
                )

            last_count = len(transcripts)

        # Current uncommitted transcript
        if transcript.text != last_text:
            print(
                f"\r[CURRENT] {transcript.text}",
                end="",
                flush=True,
            )
            last_text = transcript.text

        stop_printer.wait(0.1)
```
- `asr.get_transcript`: get the current transcirption
- `asr.get_transcripts`: get whole transcirptions

### 3. Callback setting
```py

def callback(indata, frames, time_info, status):
    if status:
        print(status)

    audio = indata[:, 0].copy()

    # Non-blocking.
    asr.feed(audio)
```
- `asr.feed`: a threading function for async to process audio for transcribing continuously

### 4. Microphone recording setting

```py

# Microphone parameters
samplerate = 16000
block_duration = 0.25
block_size = int(samplerate * block_duration)
channels = 1

# Streaming asr start
asr.start()

# Printer threading
printer_thread = threading.Thread(
    target=result_loop,
    daemon=True,
)

# Printer start
printer_thread.start()

try:
    with sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        callback=callback,
        blocksize=block_size,
        dtype="float32",
    ):
        print(
            "🎤 Recording for ASR... "
            "Press Ctrl+C to stop."
        )

        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\n⏹️ Recording stopped.")

finally:
    # Stop the live result printer first.
    stop_printer.set()
    printer_thread.join()

    # Flush the current transcript.
    end_thread = asr.end()
    end_thread.join()

    # Final result.
    transcripts = asr.get_transcripts()

    print("\n")
    print("========== Final Result ==========")

# 🎤 Recording for ASR... Press Ctrl+C to stop.
# [CURRENT]  This is voice test.
# [COMMITTED]  This is voice test.
# [CURRENT]  Can you hear me?
# [COMMITTED]  Can you hear me?
# [CURRENT] ^C
# ⏹️ Recording stopped.
#
#
# ========== Final Result ==========
# [00:00:00.050 --> 00:00:10.250]  This is voice test.
# -------------------------------
# [00:00:00.050 --> 00:00:00.050] [_BEG_]
# [00:00:00.480 --> 00:00:01.110]  This
# [00:00:01.640 --> 00:00:01.640]  is
# [00:00:01.710 --> 00:00:02.960]  voice
# [00:00:02.960 --> 00:00:03.340]  test
# [00:00:03.340 --> 00:00:08.380] .
# [00:00:08.380 --> 00:00:10.250] [_TT_150]
# -------------------------------
#
# [00:00:10.550 --> 00:00:20.750]  Can you hear me?
# -------------------------------
# [00:00:10.550 --> 00:00:10.550] [_BEG_]
# [00:00:10.930 --> 00:00:11.040]  Can
# [00:00:11.040 --> 00:00:11.480]  you
# [00:00:11.580 --> 00:00:12.190]  hear
# [00:00:12.190 --> 00:00:12.250]  me
# [00:00:12.250 --> 00:00:17.210] ?
# [00:00:17.210 --> 00:00:20.750] [_TT_100]
# -------------------------------
```

# License
This project follows [whisper.cpp](https://github.com/ggml-org/whisper.cpp/) license as MIT