"""
encoders.py — Observation encoders for pixel-based environments (Procgen).

CPU-only: pure numpy block averaging for resize (no cv2 dependency).

Default: 64×64×3 → grayscale → 32×32 → flatten to 1024-dim vector.
This lets the existing MLP-based pipeline (EpisodeLog, Monitor) work
without architecture changes.
"""

from __future__ import annotations
import numpy as np


class ProcgenEncoder:
    """Grayscale + block-average resize + flatten.

    Input:   (64, 64, 3) uint8   (Procgen default)
    Output:  (N,) float32        (N = target_size^2)
    """

    def __init__(self, target_size: int = 32):
        self.ts = target_size

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        gray = rgb.mean(axis=-1).astype(np.float32) / 255.0
        h, w = gray.shape
        bh, bw = h // self.ts, w // self.ts
        cropped = gray[:bh * self.ts, :bw * self.ts]
        downsampled = cropped.reshape(self.ts, bh, self.ts, bw).mean(axis=(1, 3))
        return downsampled.ravel().astype(np.float32)

    @property
    def obs_dim(self) -> int:
        return self.ts * self.ts