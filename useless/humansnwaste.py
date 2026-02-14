import cv2 as cv
from ultralytics import YOLO

# -------------------------------
# Geometry helpers
# -------------------------------
def inside_ratio(small, big):
    xA = max(small[0], big[0])
    yA = max(small[1], big[1])
    xB = min(small[2], big[2])
    yB = min(small[3], big[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    smallArea = (small[2] - small[0]) * (small[3] - small[1])

    if smallArea == 0:
        return 0

    return interArea / smallArea

def box_center(box):
    return ((box[0]+box[2])//2, (box[1]+box[3])//2)

def box_area(box):
    return (box[2]-box[0]) * (box[3]-box[1])


# -------------------------------
# Load image
# -------------------------------
img = cv.imread("photos/roiwastetest.jpg")
if img is None:
    print("Image not found")
    exit()

# -------------------------------
# Load models
# -------------------------------
model_general = YOLO("yolov8m.pt")   # person, couch, etc
model_waste   = YOLO("best.pt")      # your custom waste model

# -------------------------------
# Run inference
# -------------------------------
results_general = model_general(img)
results_waste   = model_waste(img)

# -------------------------------
# Collect PERSON boxes
# -------------------------------
person_boxes = []

for r in results_general:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        if model_general.names[cls_id] == "person":
            person_boxes.append(tuple(map(int, box.xyxy[0])))

# -------------------------------
# Draw GENERAL detections (green)
# -------------------------------
for r in results_general:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model_general.names[cls_id]

        cv.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv.putText(img, f"{label} {conf:.2f}", (x1, y1-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# -------------------------------
# Draw WASTE detections (red) with smart human filtering
# -------------------------------
for r in results_waste:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        waste_box = (x1, y1, x2, y2)

        skip = False

        for pbox in person_boxes:
            inside = inside_ratio(waste_box, pbox)

            if inside > 0.6:   # mostly inside a person
                wx, wy = box_center(waste_box)
                px1, py1, px2, py2 = pbox

                # Person center (face + torso region)
                px_center = (px1 + px2) // 2
                py_center = (py1 + py2) // 2

                # Distance from person center
                dist = abs(wx - px_center) + abs(wy - py_center)
                person_size = (px2 - px1) + (py2 - py1)
                norm_dist = dist / person_size

                # Waste size relative to person
                size_ratio = box_area(waste_box) / box_area(pbox)

                # Block only face/skin/clothes
                if norm_dist < 0.25 and size_ratio < 0.05:
                    skip = True
                    break

        if skip:
            continue

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model_waste.names[cls_id]

        cv.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
        cv.putText(img, f"WASTE:{label} {conf:.2f}", (x1, y2+20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# -------------------------------
# Show result
# -------------------------------
cv.imshow("Human-Aware Waste Detection", img)
cv.waitKey(0)
cv.destroyAllWindows()
