import cv2
from ultralytics import YOLO

def run_inference(image_path, model_path="best.pt", conf_threshold=0.25, save_path="result.jpg", show=False):
    """
    Runs YOLO inference, prints detections, draws boxes on the image and saves the result.
    """
    model = YOLO(model_path)
    results = model(image_path)  # runs inference

    img = cv2.imread(image_path)
    if img is None:
        print("Could not read image:", image_path)
        return

    for result in results:
        boxes = result.boxes  # Boxes object
        if boxes is None or len(boxes) == 0:
            continue

        for i in range(len(boxes)):
            # extract xyxy, confidence and class
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
            conf = float(boxes.conf[i].cpu().numpy())      # confidence
            cls_id = int(boxes.cls[i].cpu().numpy())       # class id
            cls_name = model.names.get(cls_id, str(cls_id)) if hasattr(model, "names") else str(cls_id)

            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = xyxy
            label = f"{cls_name} {conf:.2f}"
            print(label, xyxy)

            # draw box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(save_path, img)
    if show:
        cv2.imshow("detections", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference("test.jpg")
