import cv2
import os
from ultralytics import YOLO

# Get the absolute path to the model file relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "aimodel", "best.pt")
model = YOLO(model_path)



def detectwaste(frame):
    results = model(frame)
    h, w, _ = frame.shape
    print(h)

    # Draw rectangles around detected waste objects
    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Get confidence score
            confidence = box.conf[0].item()

            # Get class label
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put text label with class name and confidence
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return results
