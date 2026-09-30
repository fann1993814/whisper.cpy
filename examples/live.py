import sounddevice as sd
import threading

from whispercpy import StreamingASR, WebRTCVAD
from whispercpy.common import to_timestamp


WHISPER_CPP_PATH = "../../whisper.cpp"

lib_path = f"{WHISPER_CPP_PATH}/build/bin/libwhisper.dylib"
model_path = f"{WHISPER_CPP_PATH}/models/ggml-tiny.bin"


asr = StreamingASR(
    lib_path=lib_path,
    asr_model_path=model_path,
    language="en",
    step_ms=500,
    keep_ms=250,
    return_token=True,
    use_gpu=True,
    speech_detector=WebRTCVAD(),
)


samplerate = 16000
block_duration = 0.25
block_size = int(samplerate * block_duration)
channels = 1


# ----------------------------------------------------------------------
# Result printer
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Audio callback
# ----------------------------------------------------------------------

def callback(indata, frames, time_info, status):
    if status:
        print(status)

    audio = indata[:, 0].copy()

    # Non-blocking.
    asr.feed(audio)


# ----------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------

asr.start()

printer_thread = threading.Thread(
    target=result_loop,
    daemon=True,
)

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

    for transcript in transcripts:
        print(
            f"[{to_timestamp(transcript.t0, False)}"
            f" --> "
            f"{to_timestamp(transcript.t1, False)}] "
            f"{transcript.text}"
        )

        print("-------------------------------")

        for token in transcript.tokens:
            print(
                f"[{to_timestamp(token.t0, False)}"
                f" --> "
                f"{to_timestamp(token.t1, False)}] "
                f"{token.text}"
            )

        print("-------------------------------")
        print()

    asr.stop()
    asr.close()
