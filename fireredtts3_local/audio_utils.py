from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .paths import OUTPUTS_DIR, ensure_dirs


def load_audio(path: str):
    if not path:
        raise ValueError("Audio file is required.")
    wav, sr = sf.read(path, always_2d=True, dtype="float32")
    wav = wav[:, 0]
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    return torch.from_numpy(np.ascontiguousarray(wav)[None, :]), int(sr)


def save_tensor_audio(audio: torch.Tensor, sr: int, prefix: str) -> str:
    ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = OUTPUTS_DIR / f"{prefix}-{stamp}.wav"
    x = audio.detach().float().cpu().numpy()
    if x.ndim > 1:
        x = x[0]
    sf.write(path, np.clip(x, -1.0, 1.0), int(sr), subtype="PCM_16")
    return str(path)
