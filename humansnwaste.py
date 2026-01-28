import cv2 as cv
from ultralytics import YOLO

# Function to compute Intersection over Union (IoU) between two boxes
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / (boxAArea + boxBArea - interArea + 1e-6)


# Load image
img = cv.imread("photos/vishnunwaste3.jpeg")
if img is None:
    print("Image not found")
    exit()

# Load both models
model_general = YOLO("yolov8m.pt")   # person, dog, cat...
model_waste   = YOLO("best.pt")      # your custom waste model

# Run inference
results_general = model_general(img)
results_waste   = model_waste(img)

# -------- General model (green) --------
for result in results_general:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model_general.names[cls_id]

        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(img,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)
person_boxes = []

for r in results_general:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        if model_general.names[cls_id] == "person":
            person_boxes.append(tuple(map(int, box.xyxy[0])))

# -------- Waste model (red) --------
# for result in results_waste:
#     for box in result.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cls_id = int(box.cls[0])
#         conf = float(box.conf[0])
#         label = model_waste.names[cls_id]
#
#         cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         cv.putText(img,
#                     f"WASTE:{label} {conf:.2f}",
#                     (x1, y2 + 20),
#                     cv.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (0, 0, 255),
#                     2)

for r in results_waste:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        waste_box = (x1, y1, x2, y2)

        # Check overlap with any person
        skip = False
        for pbox in person_boxes:
            if iou(waste_box, pbox) > 0.3:  # 30% overlap
                skip = True
                break

        if skip:
            continue

        label = model_waste.names[int(box.cls[0])]
        conf = float(box.conf[0])

        cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.putText(img,
                   f"WASTE:{label} {conf:.2f}",
                   (x1, y2 + 20),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.6,
                   (0, 0, 255),
                   2)

# Show final result once
cv.imshow("Combined Detection", img)
cv.waitKey(0)
cv.destroyAllWindows()



