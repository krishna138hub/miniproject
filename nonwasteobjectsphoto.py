from ultralytics import YOLO
import cv2

# -------------------------------
# LOAD PRETRAINED COCO MODEL
# -------------------------------
# This model is trained on 80 everyday objects
model = YOLO("yolov8n.pt")

# -------------------------------
# LOAD IMAGE
# -------------------------------
image_path = "photos/bottle2.jpeg"   # change to your image file
frame = cv2.imread(image_path)

# -------------------------------
# RUN OBJECT DETECTION
# -------------------------------
results = model(frame)

# -------------------------------
# DRAW RESULTS
# -------------------------------
for result in results:
    boxes = result.boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        label = model.names[cls]

        # draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # draw label
        cv2.putText(frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

# -------------------------------
# SHOW OUTPUT
# -------------------------------
cv2.imshow("COCO Object Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

