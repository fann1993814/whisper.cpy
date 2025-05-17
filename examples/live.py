import sounddevice as sd

from whispercpy import WhipserCPP, WhisperStream
from whispercpy.utils import to_timestamp
from whispercpy.constant import STREAMING_ENDING

WHISPER_CPP_PATH = "../../whisper.cpp"

lib_path = f"{WHISPER_CPP_PATH}/build/src/libwhisper.dylib"
model_path = f"{WHISPER_CPP_PATH}/models/ggml-tiny.bin"

core = WhipserCPP(lib_path, model_path, use_gpu=True)
asr = WhisperStream(core, language='en')

samplerate = 16000
block_duration = 0.25
block_size = int(samplerate * block_duration)
channels = 1
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


def print_result():
    transcripts = asr.get_transcripts()

    for transcript in transcripts:
        print(f'[{to_timestamp(transcript.t0, False)}' +
              " --> " + f'{to_timestamp(transcript.t1, False)}] ' + transcript.text)


# Recording
try:
    with sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        callback=callback,
        blocksize=block_size,
        dtype='float32'
    ):
        print("🎤 Recording for ASR... Press Ctrl+C to stop.")
        while True:
            sd.sleep(1000)
except KeyboardInterrupt:
    print("⏹️ Recording stopped.")
    # send end signal
    asr.pipe(STREAMING_ENDING)
    print_result()
