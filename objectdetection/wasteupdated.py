import os
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "aimodel", "best.pt")

model = YOLO(model_path)


def detectwaste(frame):

    results = model.track(frame, persist=True, verbose=False)

    h, w, _ = frame.shape
    frame_area = h * w

    best_detection = None
    best_conf = 0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            if box.id is None:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            track_id = int(box.id[0])

            box_area = (x2 - x1) * (y2 - y1)

            # ❗ Reject very large objects (likely person)
            if box_area > frame_area * 0.25:
                continue

            # Keep only most confident detection
            if conf > best_conf:
                best_conf = conf
                best_detection = {
                    "id": track_id,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf
                }

    if best_detection:
        return [best_detection]
    else:
        return []


