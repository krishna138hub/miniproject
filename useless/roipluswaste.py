import cv2
from ultralytics import YOLO

# ==============================
# CONFIG
# ==============================
IMAGE_PATH = "photos/roiwastetest.jpg"
MODEL_PATH = "best.pt"
ROI_PERCENT = 0.60   # 30% of width & height (top-right corner)

# ==============================
# LOAD MODEL & IMAGE
# ==============================
model = YOLO(MODEL_PATH)
img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Image not found")
    exit()

h, w, _ = img.shape

# ==============================
# DEFINE TOP-RIGHT ROI
# ==============================
roi_w = int(w * ROI_PERCENT)
roi_h = int(h * ROI_PERCENT)

x1_roi = w - roi_w
y1_roi = 0
x2_roi = w
y2_roi = roi_h

# Draw ROI box (Blue)
cv2.rectangle(img, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 0, 0), 2)

# ==============================
# RUN YOLO
# ==============================
results = model(img)[0]

# ==============================
# PROCESS DETECTIONS
# ==============================
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cls = int(box.cls[0])
    label = model.names[cls]

    # Center of detected box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Check if center is inside ROI
    inside_roi = (x1_roi <= cx <= x2_roi) and (y1_roi <= cy <= y2_roi)

    # ❌ Skip detections inside ROI
    if inside_roi:
        continue

    # ✅ Draw only valid waste detections
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2)

# ==============================
# DISPLAY RESULT
# ==============================
cv2.imshow("Waste Detection (ROI Protected)", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
