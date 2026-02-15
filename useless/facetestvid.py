import cv2 as cv
import mediapipe as mp

# Load MediaPipe face detection
mp_face = mp.solutions.face_detection

# Long-range face detector (model_selection=1)
face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6
)

# -------------------------------
# Load VIDEO (not webcam!)
# -------------------------------
video_path = "../photos/input4.mp4"   # <-- change this
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break   # video finished

    h, w, _ = frame.shape

    # Convert BGR → RGB
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Run face detection
    results = face_detector.process(rgb)

    if results.detections:
        for det in results.detections:
            box = det.location_data.relative_bounding_box

            # Convert relative coords → pixel coords
            x1 = int(box.xmin * w)
            y1 = int(box.ymin * h)
            x2 = int((box.xmin + box.width) * w)
            y2 = int((box.ymin + box.height) * h)

            # Clip to frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Draw face box
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show frame
    cv.imshow("Long Range Face Detection", frame)

    # Press q to quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
