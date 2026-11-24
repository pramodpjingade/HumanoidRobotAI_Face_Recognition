import sounddevice as sd
import numpy as np
import webrtcvad
import collections

SAMPLE_RATE = 16000
DURATION = 5
MIC_DEVICE = 1
FRAME_DURATION = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

vad = webrtcvad.Vad(2)

print("Recording for 5 seconds...")
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

speech_frames = 0
total_frames = 0

for start in range(0, len(frame_bytes) - frame_length + 1, frame_length):
    frame = frame_bytes[start:start + frame_length]
    is_speech = vad.is_speech(frame, SAMPLE_RATE)

    total_frames += 1

    if is_speech:
        speech_frames += 1

print("\nVAD TEST RESULT")
print("Total frames:", total_frames)
print("Speech frames:", speech_frames)
print("Speech percentage:", round(speech_frames / total_frames * 100, 2), "%")