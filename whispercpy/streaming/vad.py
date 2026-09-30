import numpy as np

import webrtcvad


class WebRTCVAD:
    """
    WebRTC VAD for complete audio windows.

    Accepts mono float32 audio samples in [-1, 1] and internally
    splits them into valid WebRTC VAD frames.
    """

    VALID_SAMPLE_RATES = (8000, 16000, 32000, 48000)
    VALID_FRAME_MS = (10, 20, 30)

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 10,
        mode: int = 3,
        speech_ratio: float = 0.5,
    ):
        if sample_rate not in self.VALID_SAMPLE_RATES:
            raise ValueError(
                f"Unsupported sample_rate: {sample_rate}. "
                f"Expected one of {self.VALID_SAMPLE_RATES}."
            )

        if frame_ms not in self.VALID_FRAME_MS:
            raise ValueError(
                f"Unsupported frame_ms: {frame_ms}. "
                f"Expected one of {self.VALID_FRAME_MS}."
            )

        if not 0 <= mode <= 3:
            raise ValueError(
                "mode must be between 0 and 3."
            )

        if not 0.0 <= speech_ratio <= 1.0:
            raise ValueError(
                "speech_ratio must be between 0.0 and 1.0."
            )

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.mode = mode
        self.speech_ratio = speech_ratio

        self.frame_samples = (
            sample_rate * frame_ms // 1000
        )

        self.vad = webrtcvad.Vad(self.mode)

    def reset(self) -> None:
        """Reset VAD state."""
        pass

    def __call__(self, audio: np.ndarray) -> bool:
        """
        Determine whether the given audio window contains speech.

        Args:
            audio:
                Mono float32 audio samples in [-1.0, 1.0].

        Returns:
            True if the speech ratio reaches the configured threshold.
        """

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.ndim != 1:
            raise ValueError(
                "WebRTCVAD expects a mono audio array."
            )

        if audio.size == 0:
            return False

        audio = np.clip(audio, -1.0, 1.0)

        pcm = (
            audio * 32767.0
        ).astype(np.int16)

        n_frames = len(pcm) // self.frame_samples

        if n_frames == 0:
            return False

        speech_count = 0

        for i in range(n_frames):
            start = i * self.frame_samples
            end = start + self.frame_samples

            frame = pcm[start:end]

            if self.vad.is_speech(
                frame.tobytes(),
                self.sample_rate,
            ):
                speech_count += 1

        speech_ratio = speech_count / n_frames

        return speech_ratio >= self.speech_ratio
