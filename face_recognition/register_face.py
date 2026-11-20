
import cv2
import json
from pathlib import Path
from insightface.app import FaceAnalysis

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE = BASE_DIR / "datasets" / "face_database.json"

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

name = input("Enter the person's name: ").strip()

if not name:
    print("Name cannot be empty.")
    exit()

if DATABASE.exists():
    with open(DATABASE, "r") as file:
        database = json.load(file)
else:
    database = {}

for person, data in database.items():
    if data and isinstance(data[0], (int, float)):
        database[person] = [data]

embeddings = []

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not detected!")
    exit()

print("\nLook at the camera.")
print("Press S to capture each sample.")
print("Collect 10 samples with different angles.")
print("Press Q to cancel.")

while len(embeddings) < 10:
    success, frame = camera.read()

    if not success:
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Samples: {len(embeddings)}/10",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Advanced Face Registration", frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('s') or key == ord('S'):

        if len(faces) == 0:
            print("No face detected.")

        elif len(faces) > 1:
            print("Multiple faces detected. Keep only one person in the camera.")

        else:
            embeddings.append(faces[0].normed_embedding.tolist())
            print(f"Sample {len(embeddings)} captured.")

    elif key == ord('q') or key == ord('Q'):
        break

camera.release()
cv2.destroyAllWindows()

if len(embeddings) == 10:
    existing = database.get(name, [])
    database[name] = existing + embeddings

    DATABASE.parent.mkdir(parents=True, exist_ok=True)

    with open(DATABASE, "w") as file:
        json.dump(database, file, indent=4)

    print(f"\n{name} registered successfully!")
    print(f"Total stored embeddings: {len(database[name])}")
else:
    print("\nRegistration cancelled. Database was not changed.")
