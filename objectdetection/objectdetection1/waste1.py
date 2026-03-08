from ultralytics import YOLO
import gc

model = YOLO("yolov8s.pt")

target_objects = ["bottle", "remote","frisbee","tennis racket","bowl","box","vase","book","skateboard"]
conf_threshold = 0.1
img_size = 960

track_id = 0


def detectwaste(frame):

    global track_id

    results = model(frame, conf=conf_threshold, imgsz=img_size)[0]

    waste_boxes = []

    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        class_name = model.names[cls]

        if class_name in target_objects:

            waste_boxes.append({
                "id": track_id,
                "bbox": (x1, y1, x2, y2),
                "label": class_name,
                "conf": conf
            })
            print("Detected:", class_name, "Confidence:", conf)
            print('TRACK ID:', track_id)
            track_id = 0  # Only one object tracked
    print("waste_boxes:", waste_boxes)
    del results
    gc.collect()
    return waste_boxes
