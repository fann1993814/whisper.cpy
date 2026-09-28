from __future__ import annotations

import numpy as np


class SlidingAudioBuffer:
    """
    Audio buffer for streaming ASR.

    The buffer maintains:
        - old audio: audio retained from the previous inference window
        - new audio: newly received audio waiting to be processed
        - window: current inference window

    Once enough new audio has accumulated, a sliding window is built from:

        retained old audio + new audio

    This preserves the behavior of the original WhisperStream implementation.
    """

    def __init__(
        self,
        step_samples: int,
        keep_samples: int,
        length_samples: int,
    ) -> None:
        if step_samples <= 0:
            raise ValueError("step_samples must be positive")

        if keep_samples < 0:
            raise ValueError("keep_samples must be non-negative")

        if length_samples <= 0:
            raise ValueError("length_samples must be positive")

        # Preserve the original WhisperStream behavior:
        #
        # keep_ms = min(keep_ms, step_ms)
        # length_ms = max(length_ms, step_ms)
        keep_samples = min(keep_samples, step_samples)
        length_samples = max(length_samples, step_samples)

        self.step_samples = step_samples
        self.keep_samples = keep_samples
        self.length_samples = length_samples

        # Correspond to the original:
        #
        # pcmf32_old
        # pcmf32_new
        # pcmf32
        self._old = np.empty(0, dtype=np.float32)
        self._new = np.empty(0, dtype=np.float32)
        self._window = np.empty(0, dtype=np.float32)

        # Metadata for the most recently built window.
        #
        # _old_samples_before_build is important because _old is
        # replaced by _window after build_window().
        self._old_samples_before_build = 0
        self._overlap_samples = 0

    def append(self, audio: np.ndarray) -> None:
        """Append new audio samples."""

        audio = np.ascontiguousarray(audio, dtype=np.float32)

        if audio.size == 0:
            return

        if self._new.size == 0:
            self._new = audio.copy()
        else:
            self._new = np.concatenate(
                (self._new, audio)
            )

    def ready(self) -> bool:
        """
        Return whether enough new audio is available for processing.

        This corresponds to the original:

            if len(pcmf32_new) >= n_samples_step:
        """

        return len(self._new) >= self.step_samples

    def build_window(self) -> np.ndarray:
        """
        Build the current inference window.

        The original implementation calculates:

            n_samples_take = min(
                len(pcmf32_old),
                max(
                    0,
                    n_samples_keep + n_samples_len - n_samples_new
                )
            )

        and then builds:

            pcmf32 = pcmf32_old[-n_samples_take:] + pcmf32_new
        """

        if not self.ready():
            raise RuntimeError(
                "Audio buffer is not ready for processing"
            )

        n_samples_new = len(self._new)

        # IMPORTANT:
        # Save the size before replacing _old with the new window.
        old_samples_before_build = len(self._old)
        self._old_samples_before_build = old_samples_before_build

        n_samples_take = min(
            old_samples_before_build,
            max(
                0,
                self.keep_samples
                + self.length_samples
                - n_samples_new,
            ),
        )

        self._overlap_samples = n_samples_take

        if n_samples_take > 0:
            old_audio = self._old[-n_samples_take:]
            self._window = np.concatenate(
                (old_audio, self._new)
            )
        else:
            self._window = self._new.copy()

        # Same as the original:
        #
        # pcmf32_old = pcmf32
        #
        # The entire current inference window becomes the "old"
        # audio for the next iteration.
        self._old = self._window.copy()

        # New audio has now been consumed into the current window.
        self._new = np.empty(
            0,
            dtype=np.float32,
        )

        return self._window

    def get_window(self) -> np.ndarray:
        """Return the most recently built inference window."""

        return self._window

    def keep_tail(self) -> None:
        """
        Keep the last keep_samples from the current window.

        This corresponds to the original flush():

            pcmf32_old = pcmf32[-n_samples_keep:]
        """

        if self.keep_samples <= 0:
            self._old = np.empty(
                0,
                dtype=np.float32,
            )
            return

        self._old = self._window[-self.keep_samples:].copy()

    def clear(self) -> None:
        """Clear all buffered audio."""

        self._old = np.empty(
            0,
            dtype=np.float32,
        )
        self._new = np.empty(
            0,
            dtype=np.float32,
        )
        self._window = np.empty(
            0,
            dtype=np.float32,
        )

        self._old_samples_before_build = 0
        self._overlap_samples = 0

    @property
    def new_samples(self) -> int:
        return len(self._new)

    @property
    def window_samples(self) -> int:
        return len(self._window)

    @property
    def old_samples(self) -> int:
        return len(self._old)

    @property
    def overlap_samples(self) -> int:
        """
        Number of old samples actually preserved when building
        the most recent inference window.
        """

        return self._overlap_samples

    @property
    def overlap_preserved(self) -> bool:
        """
        Whether the entire previous old buffer was preserved.

        This corresponds exactly to the original:

            if len(pcmf32_old) > n_samples_take:
                prev_inference_overlap_ms = 0

        Therefore:

            overlap_preserved = (
                n_samples_take >= len(pcmf32_old)
            )
        """

        return (
            self._overlap_samples
            >= self._old_samples_before_build
        )
