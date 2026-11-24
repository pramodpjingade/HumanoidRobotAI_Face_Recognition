# Humanoid AI – Face Recognition System

An AI-powered facial recognition system developed for humanoid robots to detect, recognize, and identify people in real time using computer vision and deep learning.

## Features

- Real-time face detection using a webcam
- Face recognition and identity identification
- Face registration for new users
- Face embedding generation and comparison
- Recognition logging
- Unknown person detection
- Integration-ready architecture for humanoid robotics

## Technologies Used

- Python 3.11
- OpenCV
- InsightFace
- NumPy
- ONNX Runtime
- Deep Learning
- Computer Vision

## Project Structure

```text
HumanoidAI-Face-Recognition/
│
├── face_recognition/
│   ├── face_test.py
│   ├── recognition_log.py
│   ├── recognize_face.py
│   ├── register_face.py
│   └── webcam_face_detection.py
│
├── datasets/
│   └── Face recognition database
│
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Register a person's face using the registration script.
2. Run the face recognition module.
3. Allow webcam access.
4. The system detects faces and identifies registered people.
5. Recognition results are recorded in the log.

Run the relevant scripts from the project directory:

```bash
python face_recognition/register_face.py
```

```bash
python face_recognition/recognize_face.py
```

## Application

This project is designed as a face identification module for a humanoid robot. It can later work alongside voice recognition and multimodal identity fusion to improve human–robot interaction.

## Future Improvements

- Multi-person tracking
- Improved recognition under different lighting conditions
- Face anti-spoofing
- Real-time identity fusion with voice recognition
- Integration with an AI voice assistant
- Humanoid robot interaction

## Privacy

Personal face images, biometric embeddings, private datasets, and recognition logs should not be uploaded publicly without permission.

## Project Status

Under development as part of a Humanoid Robotics and Artificial Intelligence project.
