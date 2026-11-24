
import sounddevice as sd
from scipy.io.wavfile import write
from pathlib import Path
import numpy as np

SAMPLE_RATE = 16000
DURATION = 8
DEVICE_ID = 1
PERSON_NAME = "PRAMOD P JINGADE"

SAVE_DIR = Path("datasets/voice_samples") / PERSON_NAME
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("VOICE REGISTRATION")
print("Speaker:", PERSON_NAME)
print("Samples required: 5")

for i in range(1, 6):
    input(f"\nPress ENTER to record sample {i}/5...")

    print("Recording started. Speak naturally...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=DEVICE_ID
    )

    sd.wait()

    filename = SAVE_DIR / f"sample_{i}.wav"
    write(str(filename), SAMPLE_RATE, audio)

    print("Saved:", filename)

print("\nAll five voice samples recorded successfully!")
