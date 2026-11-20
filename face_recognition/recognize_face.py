
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

MAX_MISSED_FRAMES = 10
IOU_THRESHOLD = 0.25

tracks = {}
next_track_id = 0


def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


def create_track(box):

    return {
        "bbox": box,
        "candidate_name": None,
        "candidate_count": 0,
        "confirmed_name": None,
        "logged_name": None,
        "missed": 0
    }


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not detected!")
    exit()

print("Multi-Person Face Recognition Started!")
print("Press Q to exit.")

while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    faces = app.get(frame)

    detections = []

    for face in faces:

        x1, y1, x2, y2 = map(int, face.bbox)

        box = (x1, y1, x2, y2)

        detections.append({
            "face": face,
            "bbox": box
        })

    # Match current faces with previous face tracks

    matches = []

    for track_id, track in tracks.items():

        for index, detection in enumerate(detections):

            iou = calculate_iou(
                track["bbox"],
                detection["bbox"]
            )

            if iou >= IOU_THRESHOLD:
                matches.append((iou, track_id, index))

    matches.sort(reverse=True)

    assigned_tracks = set()
    assigned_detections = {}

    for iou, track_id, index in matches:

        if track_id not in assigned_tracks and index not in assigned_detections:

            assigned_tracks.add(track_id)
            assigned_detections[index] = track_id

    # Update missed frames

    for track_id in list(tracks.keys()):

        if track_id not in assigned_tracks:
            tracks[track_id]["missed"] += 1

        else:
            tracks[track_id]["missed"] = 0

    # Remove old tracks

    for track_id in list(tracks.keys()):

        if tracks[track_id]["missed"] > MAX_MISSED_FRAMES:
            del tracks[track_id]

    # Create tracks for new faces

    for index, detection in enumerate(detections):

        if index not in assigned_detections:

            tracks[next_track_id] = create_track(
                detection["bbox"]
            )

            assigned_detections[index] = next_track_id

            next_track_id += 1

    # Recognize every face independently

    for index, detection in enumerate(detections):

        face = detection["face"]
        x1, y1, x2, y2 = detection["bbox"]

        track_id = assigned_detections[index]
        track = tracks[track_id]

        face_width = x2 - x1
        face_height = y2 - y1

        if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:

            track["candidate_name"] = None
            track["candidate_count"] = 0
            track["confirmed_name"] = None

            display_name = "Face Too Small"
            color = (0, 165, 255)

        else:

            height, width = frame.shape[:2]

            x1_crop = max(0, x1)
            y1_crop = max(0, y1)
            x2_crop = min(width, x2)
            y2_crop = min(height, y2)

            face_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]

            gray_face = cv2.cvtColor(
                face_crop,
                cv2.COLOR_BGR2GRAY
            )

            blur_score = cv2.Laplacian(
                gray_face,
                cv2.CV_64F
            ).var()

            if blur_score < MIN_BLUR_SCORE:

                track["candidate_name"] = None
                track["candidate_count"] = 0
                track["confirmed_name"] = None

                display_name = "Blurry Face"
                color = (0, 165, 255)

            else:

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

                    track["candidate_name"] = None
                    track["candidate_count"] = 0
                    track["confirmed_name"] = None
                    track["logged_name"] = None

                    display_name = "Unknown Person"
                    color = (0, 0, 255)

                else:

                    if track["candidate_name"] == best_name:

                        track["candidate_count"] += 1

                    else:

                        track["candidate_name"] = best_name
                        track["candidate_count"] = 1
                        track["confirmed_name"] = None
                        track["logged_name"] = None

                    if track["candidate_count"] >= REQUIRED_FRAMES:

                        track["confirmed_name"] = best_name

                    if (
                        track["confirmed_name"] == best_name
                        and track["logged_name"] != best_name
                    ):

                        save_recognition(best_name, best_score)
                        track["logged_name"] = best_name

                    if track["confirmed_name"]:

                        display_name = track["confirmed_name"]
                        color = (0, 255, 0)

                    else:

                        display_name = "Verifying..."
                        color = (0, 165, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            f"{display_name} | ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

    cv2.imshow("Multi-Person Humanoid Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        break

camera.release()
cv2.destroyAllWindows()
