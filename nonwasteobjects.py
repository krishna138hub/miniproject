from ultralytics import YOLO
import cv2

# ================================
# SELECT IMPROVEMENT OPTIONS
# ================================

USE_LARGER_MODEL = True
LOWER_CONFIDENCE = True
HIGH_RESOLUTION = True
ZOOM_FRAME = False
IMPROVE_LIGHTING = False

# ================================
# LOAD MODEL
# ================================

if USE_LARGER_MODEL:
    model = YOLO("yolov8s.pt")
else:
    model = YOLO("yolov8n.pt")

# ================================
# VIDEO INPUT
# ================================

video_path = "photos/input4.mp4"   # change to 0 for webcam
cap = cv2.VideoCapture(video_path)

# confidence threshold
conf_threshold = 0.5
if LOWER_CONFIDENCE:
    conf_threshold = 0.25

# image size
img_size = 640
if HIGH_RESOLUTION:
    img_size = 960

# detect only these objects
target_objects = ["bottle", "remote"]

# ================================
# MAIN LOOP
# ================================

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # ----------------------------
    # Solution 4: Zoom frame
    # ----------------------------
    if ZOOM_FRAME:
        frame = cv2.resize(frame, None, fx=1.5, fy=1.5)

    # ----------------------------
    # Solution 5: Improve lighting
    # ----------------------------
    if IMPROVE_LIGHTING:
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)

    # ----------------------------
    # Run YOLO detection
    # ----------------------------
    results = model(frame, conf=conf_threshold, imgsz=img_size)[0]

    # ----------------------------
    # Draw bounding boxes
    # ----------------------------
    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        class_name = model.names[cls]

        # Detect only bottle and remote
        if class_name in target_objects:

            label = f"{class_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # ----------------------------
    # Resize window so full frame fits screen
    # ----------------------------
    screen_w = 1280
    screen_h = 720

    h, w = frame.shape[:2]

    if w > screen_w or h > screen_h:
        scale = min(screen_w/w, screen_h/h)
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

    cv2.imshow("COCO Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()








