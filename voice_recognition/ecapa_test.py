
import torch
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd
from speechbrain.inference.speaker import EncoderClassifier

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Loading ECAPA-TDNN...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)

print("Model loaded!")
print("Model device:", next(classifier.mods.parameters()).device)

sample_rate, audio = wavfile.read("voice_test.wav")

if audio.ndim > 1:
    audio = audio.mean(axis=1)

if np.issubdtype(audio.dtype, np.integer):
    audio = audio.astype(np.float32) / max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
else:
    audio = audio.astype(np.float32)

if sample_rate != 16000:
    factor = gcd(sample_rate, 16000)
    audio = resample_poly(
        audio, 16000 // factor, sample_rate // factor
    ).astype(np.float32)

signal = torch.tensor(audio).unsqueeze(0).to(device)

embedding = classifier.encode_batch(signal)

print("Voice embedding generated successfully!")
print("Embedding shape:", embedding.shape)
print("Embedding device:", embedding.device)
