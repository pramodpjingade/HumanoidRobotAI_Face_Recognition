
import torch
import sounddevice as sd
import numpy as np
import json
import webrtcvad

from pathlib import Path
from scipy.signal import resample_poly
from math import gcd
from speechbrain.inference.speaker import EncoderClassifier

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 16000
DURATION = 5
MIC_DEVICE = 1
FRAME_DURATION = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

THRESHOLD = 0.40
MIN_SPEECH_PERCENT = 30

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

vad = webrtcvad.Vad(2)

input("\nPress ENTER to start recognition...")

print("\nSpeak naturally for 5 seconds...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16",
    device=MIC_DEVICE
)

sd.wait()

audio = audio.flatten()

frame_bytes = audio.tobytes()
frame_length = FRAME_SIZE * 2

speech_segments = []
speech_frames = 0
total_frames = 0

for start in range(0, len(frame_bytes) - frame_length + 1, frame_length):
    frame = frame_bytes[start:start + frame_length]

    total_frames += 1

    if vad.is_speech(frame, SAMPLE_RATE):
        speech_frames += 1
        speech_segments.append(
            np.frombuffer(frame, dtype=np.int16)
        )

speech_percentage = speech_frames / max(total_frames, 1) * 100

print("\nSpeech percentage:", round(speech_percentage, 2), "%")

if speech_percentage < MIN_SPEECH_PERCENT:
    print("Not enough speech detected. Try again.")
    raise SystemExit

if not speech_segments:
    print("No speech detected.")
    raise SystemExit

audio = np.concatenate(speech_segments).astype(np.float32) / 32768.0

if len(audio) < SAMPLE_RATE * 2:
    print("Speech duration too short. Please speak longer.")
    raise SystemExit

if np.max(np.abs(audio)) < 0.005:
    print("Audio volume too low. Try again.")
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

print("\n----- ADVANCED VOICE RECOGNITION -----")
print("Cosine similarity:", round(best_score, 4))

if best_score >= THRESHOLD:
    print("Recognized speaker:", best_person)
else:
    print("Unknown speaker detected!")

print("\nThis is a similarity score, not a probability.")
