
import cv2
from insightface.app import FaceAnalysis

print("Loading face recognition model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

print("AI model ready!")
print("Starting webcam...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam not detected!")
    exit()

while True:
    success, frame = camera.read()

    if not success:
        print("Unable to capture video.")
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face Detected",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Humanoid Robot - AI Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
