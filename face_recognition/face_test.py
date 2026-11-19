
from insightface.app import FaceAnalysis

print("Loading AI face recognition model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

print("Face recognition model loaded successfully!")
