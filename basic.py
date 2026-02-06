import cv2 as cv
from ultralytics import YOLO
img= cv.imread('photos/millie.jpg')


# Load YOLOv8 pretrained model
model = YOLO('yolov8m.pt')   # nano model (fast, good for students)


# Run detection
results = model(img)


# Draw detections
for result in results:
   for box in result.boxes:
       x1, y1, x2, y2 = map(int, box.xyxy[0])
       cls_id = int(box.cls[0])
       conf = float(box.conf[0])
       label = model.names[cls_id]
       print(label)


       # Draw rectangle
       cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


       # Put label
       cv.putText(
           img,
           f"{label} {conf:.2f}",
           (x1, y1 - 10),
           cv.FONT_HERSHEY_SIMPLEX,
           0.6,
           (0, 255, 0),
           2
       )


       # Show result
       cv.imshow('YOLO Detection', img)
       cv.waitKey(0)
       cv.destroyAllWindows()