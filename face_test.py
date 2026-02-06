import cv2 as cv
import mediapipe as mp

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

img = cv.imread("photos/millie.jpg")
if img is None:
    print("Image not found")
    exit()

h, w, _ = img.shape
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

results = face_detector.process(rgb)

if results.detections:
    for det in results.detections:
        box = det.location_data.relative_bounding_box
        x1 = int(box.xmin * w)
        y1 = int(box.ymin * h)
        x2 = int((box.xmin + box.width) * w)
        y2 = int((box.ymin + box.height))

