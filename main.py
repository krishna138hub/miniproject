import cv2
from objectdetection.hands import detecthands
from objectdetection.waste import detectwaste
from objectdetection.waste_tracking import WasteTracker
from objectdetection.face import detectface


def main():
    print("Application to detect hand and waste with littering detection")
    print("Press ESC to exit\n")

    # Initialize waste tracker
    # separation_threshold: pixel distance to consider hand and waste as separated (default: 50)
    # littering_time_threshold: seconds before marking waste as littered (default: 2.0)
    waste_tracker = WasteTracker(separation_threshold=80, littering_time_threshold=2.0)

    # Video input (change to 0 for webcam)
    url = "http://192.168.0.6:8080/video"
    cap = cv2.VideoCapture(url)


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect hands and waste
        hand_results = detecthands(frame)
        face_results = detectface(frame)
        waste_results = detectwaste(frame)

        # Update waste tracking and detect littering
        tracking_data = waste_tracker.update(hand_results, waste_results, frame)

        # Draw waste status on frame
        frame = waste_tracker.draw_waste_status(frame, tracking_data)

        # Print littered waste information
        if tracking_data['littered_waste']:
            for littered_item in tracking_data['littered_waste']:
                print(f"\n⚠️  LITTERED! Waste #{littered_item['waste_id']} has been thrown!")
                print(f"   Separated for {littered_item['time_separated']:.2f} seconds")

        height, width = frame.shape[:2]

        scale = 0.7  # reduce to 70%
        new_width = int(width * scale)
        new_height = int(height * scale)

        resized = cv2.resize(frame, (new_width, new_height))


        cv2.imshow("Hand and Waste Detection with Littering Detection", resized)

        if cv2.waitKey(1) & 0xFF == 27:
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()