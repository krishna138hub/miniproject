import cv2 as cv
import mediapipe as mp

mp_face = mp.solutions.face_detection

# Initialize detector
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

img = cv.imread("../photos/millie.jpg")
if img is None:
    print("Image not found")
    exit()

h, w, _ = img.shape
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

results = face_detector.process(rgb)

if results.detections:
    for det in results.detections:
        box = det.location_data.relative_bounding_box

        # Convert relative box to absolute pixel coordinates
        x1 = int(box.xmin * w)
        y1 = int(box.ymin * h)
        x2 = int((box.xmin + box.width) * w)
        y2 = int((box.ymin + box.height) * h)  # <-- FIXED

        # Clip values to image boundaries (very important!)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Draw rectangle
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show output
cv.imshow("Face Detection", img)

print("Press 'q' to close the window")

while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
