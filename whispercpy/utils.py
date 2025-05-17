import threading
import numpy as np
import scipy.signal as signal

from functools import wraps


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


def is_speech(frame, sample_rate=16000, energy_thold=0.01, freq_thold=100.0):
    """
    :param frame: 1D numpy float32 array (-1.0 ~ 1.0)
    :param sample_rate: e.g., 16000
    :param energy_thold: energy threshold ratio
    :param freq_thold: high-pass filter threshold (Hz)
    :return: True if speech detected, False if silence
    """

    def high_pass_filter(pcmf32, cutoff_freq, sample_rate):
        """
        High-pass filter to remove low-frequency noise
        """
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(1, normal_cutoff, btype='high', analog=False)
        return signal.lfilter(b, a, pcmf32)

    n_samples = len(frame)
    if n_samples == 0:
        return False  # No samples, assume silence

    # High-pass filter if needed
    if freq_thold > 0.0:
        frame = high_pass_filter(frame, freq_thold, sample_rate)

    # RMS
    energy_all = np.sqrt(np.mean(frame ** 2))

    return energy_all > energy_thold


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

        return thread

    return wrapper
