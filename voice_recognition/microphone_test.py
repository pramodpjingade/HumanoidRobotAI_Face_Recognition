
import sounddevice as sd
from scipy.io.wavfile import write

DURATION = 5
SAMPLE_RATE = 16000
DEVICE_ID = 1

print("Recording started...")
print("Speak something for 5 seconds.")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16",
    device=DEVICE_ID
)

sd.wait()

write("voice_test.wav", SAMPLE_RATE, audio)

print("Recording completed!")
print("Audio saved as voice_test.wav")
