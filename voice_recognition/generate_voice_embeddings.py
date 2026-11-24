import torch
import numpy as np
import json
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd
from speechbrain.inference.speaker import EncoderClassifier

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/spkrec-ecapa-voxceleb"
SAMPLE_DIR = Path("datasets/voice_samples/PRAMOD")
OUTPUT_FILE = Path("datasets/voice_database.json")

print("Loading ECAPA-TDNN model...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=MODEL_PATH,
    run_opts={"device": DEVICE}
)

print("Model loaded on:", DEVICE)

voice_database = {"PRAMOD": []}

for audio_file in sorted(SAMPLE_DIR.glob("*.wav")):

    sample_rate, audio = wavfile.read(str(audio_file))

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = max(abs(info.min), info.max)
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)

    if sample_rate != 16000:
        factor = gcd(sample_rate, 16000)
        audio = resample_poly(
            audio,
            16000 // factor,
            sample_rate // factor
        ).astype(np.float32)

    signal = torch.tensor(audio).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = classifier.encode_batch(signal)

    embedding = embedding.squeeze().cpu().numpy()

    voice_database["PRAMOD"].append(embedding.tolist())

    print(f"Processed: {audio_file.name}")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w") as file:
    json.dump(voice_database, file)

print("\nVoice embedding generation completed!")
print("Total embeddings:", len(voice_database["PRAMOD"]))
print("Embedding dimensions:", len(voice_database["PRAMOD"][0]))
print("Saved to:", OUTPUT_FILE)