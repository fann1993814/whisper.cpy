import threading
import numpy as np

from functools import wraps

from .constant import WHISPER_SAMPLE_RATE


def to_timestamp(t: int, comma: bool):
    msec = int(t * 10)
    hours = int(msec / (1000 * 60 * 60))
    msec = int(msec - hours * (1000 * 60 * 60))
    minutes = int(msec / (1000 * 60))
    msec = int(msec - minutes * (1000 * 60))
    sec = int(msec / 1000)
    msec = int(msec - sec * 1000)

    return "{:02d}:{:02d}:{:02d}{}{:03d}".format(
        hours, minutes, sec, "," if comma else ".", msec)


def is_silent(audio: np.ndarray, threshold: float = 0.01):
    if len(audio) < 2 * WHISPER_SAMPLE_RATE:
        return True
    return np.mean(np.abs(audio[-WHISPER_SAMPLE_RATE:])) < threshold


def run_aysnc(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_thread_lock"):
            self._thread_lock = threading.Lock()

        if not hasattr(self, "_thread_join"):
            self._thread_join = False

        def thread_target():
            with self._thread_lock:
                method(self, *args, **kwargs)

        thread = threading.Thread(target=thread_target)
        thread.start()

        if self._thread_join:
            thread.join()

        return

    return wrapper
