
import cv2

# Open the default webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Camera not detected!")
    exit()

print("Camera connected successfully!")
print("Press Q to close the camera.")

while True:
    success, frame = camera.read()

    if not success:
        print("Unable to capture video.")
        break

    cv2.imshow("Humanoid Robot - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
