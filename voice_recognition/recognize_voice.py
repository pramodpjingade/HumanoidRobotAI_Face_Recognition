import torch
import sounddevice as sd
import numpy as np
import json

from pathlib import Path
from scipy.signal import resample_poly
from math import gcd
from speechbrain.inference.speaker import EncoderClassifier

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 16000
DURATION = 5
MIC_DEVICE = 1

DATABASE = Path("datasets/voice_database.json")

print("Loading voice database...")

with open(DATABASE, "r") as file:
    voice_database = json.load(file)

print("Loading ECAPA-TDNN model...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

print("Model loaded on:", DEVICE)

input("\nPress ENTER to start voice recognition...")

print("\nSpeak for 5 seconds...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=MIC_DEVICE
)

sd.wait()

audio = audio.flatten()

if np.max(np.abs(audio)) < 0.001:
    print("No usable voice detected. Please try again.")
    raise SystemExit

signal = torch.tensor(audio).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    live_embedding = classifier.encode_batch(signal)

live_embedding = live_embedding.squeeze()

best_score = -1
best_person = None

for person, embeddings in voice_database.items():

    for stored_embedding in embeddings:

        stored = torch.tensor(
            stored_embedding,
            dtype=torch.float32,
            device=DEVICE
        )

        score = torch.nn.functional.cosine_similarity(
            live_embedding.unsqueeze(0),
            stored.unsqueeze(0)
        ).item()

        if score > best_score:
            best_score = score
            best_person = person

THRESHOLD = 0.40

print("\n----- VOICE RECOGNITION RESULT -----")
print("Cosine similarity:", round(best_score, 4))

if best_score >= THRESHOLD:
    print("Recognized speaker:", best_person)
else:
    print("Unknown speaker detected!")

print("\nThis is a similarity score, not a probability.")