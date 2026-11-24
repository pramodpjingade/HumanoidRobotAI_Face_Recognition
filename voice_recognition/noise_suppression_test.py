
import sys
import types
import torch
import torchaudio
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd

# Compatibility fix for DeepFilterNet 0.5.6
backend = types.ModuleType("torchaudio.backend")
backend.__path__ = []

common = types.ModuleType("torchaudio.backend.common")
common.AudioMetaData = getattr(torchaudio, "AudioMetaData", object)

backend.common = common
sys.modules["torchaudio.backend"] = backend
sys.modules["torchaudio.backend.common"] = common
torchaudio.backend = backend

from df.enhance import init_df, enhance

INPUT_FILE = Path("voice_test.wav")
OUTPUT_FILE = Path("voice_test_clean.wav")

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Recording not found: {INPUT_FILE}")

print("Loading DeepFilterNet3...")
model, df_state, _ = init_df()

print("Loading original recording...")

sample_rate, audio = wavfile.read(str(INPUT_FILE))

if audio.ndim > 1:
    audio = audio.mean(axis=1)

if np.issubdtype(audio.dtype, np.integer):
    info = np.iinfo(audio.dtype)
    audio = audio.astype(np.float32) / max(abs(info.min), info.max)
else:
    audio = audio.astype(np.float32)

if sample_rate != df_state.sr():
    factor = gcd(sample_rate, df_state.sr())
    audio = resample_poly(
        audio,
        df_state.sr() // factor,
        sample_rate // factor
    ).astype(np.float32)

signal = torch.from_numpy(audio).unsqueeze(0)

print("Applying noise suppression...")

with torch.no_grad():
    cleaned = enhance(model, df_state, signal)

cleaned = cleaned.squeeze().cpu().numpy()

cleaned = np.clip(cleaned, -1.0, 1.0)
cleaned = (cleaned * 32767).astype(np.int16)

wavfile.write(str(OUTPUT_FILE), df_state.sr(), cleaned)

print("\nNoise suppression completed!")
print("Saved:", OUTPUT_FILE)
