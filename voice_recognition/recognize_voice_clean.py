
import sys
import types
import torch
import torchaudio
import sounddevice as sd
import numpy as np
import json
import webrtcvad

from pathlib import Path
from scipy.signal import resample_poly
from math import gcd
from speechbrain.inference.speaker import EncoderClassifier

# DeepFilterNet compatibility
backend = types.ModuleType("torchaudio.backend")
backend.__path__ = []

common = types.ModuleType("torchaudio.backend.common")
common.AudioMetaData = getattr(torchaudio, "AudioMetaData", object)

backend.common = common
sys.modules["torchaudio.backend"] = backend
sys.modules["torchaudio.backend.common"] = common
torchaudio.backend = backend

from df.enhance import init_df, enhance

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

print("Loading ECAPA-TDNN...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

print("Loading DeepFilterNet3...")

df_model, df_state, _ = init_df()

print("Models loaded on:", DEVICE)

vad = webrtcvad.Vad(2)

input("\nPress ENTER to start recognition...")

print("\nRecording... Speak naturally.")

recording = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=MIC_DEVICE
)

sd.wait()

audio = recording.flatten()

print("Applying noise suppression...")

factor = gcd(SAMPLE_RATE, df_state.sr())

enhanced_audio = resample_poly(
    audio,
    df_state.sr() // factor,
    SAMPLE_RATE // factor
).astype(np.float32)

signal = torch.from_numpy(enhanced_audio).unsqueeze(0)

with torch.no_grad():
    cleaned = enhance(df_model, df_state, signal)

cleaned = cleaned.squeeze().cpu().numpy()

cleaned = resample_poly(
    cleaned,
    SAMPLE_RATE,
    df_state.sr()
).astype(np.float32)

cleaned = np.clip(cleaned, -1.0, 1.0)
print("\nComparing original and cleaned audio...")

raw_signal = torch.from_numpy(
    recording.flatten().astype(np.float32)
).unsqueeze(0).to(DEVICE)

clean_signal = torch.from_numpy(
    cleaned.astype(np.float32)
).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    raw_embedding = classifier.encode_batch(raw_signal)
    clean_embedding = classifier.encode_batch(clean_signal)

def get_best_score(embedding):
    best = -1

    for embeddings in voice_database.values():
        for stored_embedding in embeddings:
            stored = torch.tensor(
                stored_embedding,
                dtype=torch.float32,
                device=DEVICE
            )

            score = torch.nn.functional.cosine_similarity(
                embedding.reshape(1, -1),
                stored.reshape(1, -1)
            ).item()

            best = max(best, score)

    return best

print("Original audio score:", round(get_best_score(raw_embedding), 4))
print("Cleaned audio score:", round(get_best_score(clean_embedding), 4))

print("Noise suppression completed.")

audio_int16 = (cleaned * 32767).astype(np.int16)

frame_bytes = audio_int16.tobytes()
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

speech_duration = len(speech_segments) * FRAME_SIZE / SAMPLE_RATE

if speech_duration < 2:
    print("Speech duration too short. Please speak longer.")
    raise SystemExit

if np.max(np.abs(cleaned)) < 0.005:
    print("Audio volume too low. Try again.")
    raise SystemExit

signal = torch.from_numpy(
    cleaned.astype(np.float32)
).unsqueeze(0).to(DEVICE)

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

print("\n----- CLEAN VOICE RECOGNITION -----")
print("Cosine similarity:", round(best_score, 4))

if best_score >= THRESHOLD:
    print("Recognized speaker:", best_person)
else:
    print("Unknown speaker detected!")

print("\nThis is a similarity score, not a probability.")
