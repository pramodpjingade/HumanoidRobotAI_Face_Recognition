import csv
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "datasets" / "recognition_log.csv"

def save_recognition(name, score):
    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Date", "Time", "Person", "Similarity Score"])

        now = datetime.now()

        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            name,
            round(score, 3)
        ])
        