
import cv2
import json
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
from recognition_log import save_recognition

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE = BASE_DIR / "datasets" / "face_database.json"

with open(DATABASE, "r") as file:
    database = json.load(file)

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)


app.prepare(ctx_id=0, det_size=(640, 640))

THRESHOLD = 0.45
REQUIRED_FRAMES = 5

MIN_FACE_SIZE = 100
MIN_BLUR_SCORE = 30

candidate_name = None
candidate_count = 0
confirmed_name = None

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not detected!")
    exit()

print("Stable Face Recognition Started!")
print("Press Q to exit.")

while True:
    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    faces = app.get(frame)

    if not faces:
        candidate_name = None
        candidate_count = 0
        confirmed_name = None

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        face_width = x2 - x1
        face_height = y2 - y1

        if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
            continue

        face_crop = frame[y1:y2, x1:x2]

        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()

        if blur_score < MIN_BLUR_SCORE:
            continue

        embedding = face.normed_embedding
        best_name = "Unknown Person"
        best_score = -1

        for name, saved_embeddings in database.items():

            if saved_embeddings and isinstance(saved_embeddings[0], (int, float)):
                saved_embeddings = [saved_embeddings]

            for saved_embedding in saved_embeddings:
                saved_embedding = np.array(saved_embedding)
                score = float(np.dot(embedding, saved_embedding))

                if score > best_score:
                    best_score = score
                    best_name = name

        if best_score < THRESHOLD:
            best_name = "Unknown Person"

        if best_name == "Unknown Person":
            candidate_name = None
            candidate_count = 0
            confirmed_name = None

            display_name = "Unknown Person"

        else:
            if candidate_name == best_name:
                candidate_count += 1
            else:
                candidate_name = best_name
                candidate_count = 1
                confirmed_name = None

            if candidate_count >= REQUIRED_FRAMES:
                confirmed_name = best_name
            if confirmed_name == best_name and candidate_count == REQUIRED_FRAMES:
                save_recognition(best_name, best_score)

            if confirmed_name == best_name and candidate_count == REQUIRED_FRAMES:
                save_recognition(best_name, best_score)

            display_name = confirmed_name if confirmed_name else "Verifying..."

        color = (0, 255, 0) if confirmed_name else (0, 165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            f"{display_name} ({best_score:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

    cv2.imshow("Stable Humanoid Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
