import cv2
from ultralytics import YOLO

def run_video_inference(video_path):
    # Load COCO pretrained model
    model = YOLO("yolov8n.pt")   # COCO model (80 classes)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame, conf=0.3)

        # Get annotated frame (draws all detected objects)
        annotated_frame = results[0].plot()

        # Show output
        cv2.imshow("COCO Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video_inference("photos/review1.mp4")  # replace with your video file


