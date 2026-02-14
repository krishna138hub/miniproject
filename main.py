import cv2
from objectdetection.hands import detecthands
from objectdetection.waste import detectwaste
from objectdetection.waste_tracking import WasteTracker


def main():
    print("Application to detect hand and waste with littering detection")
    print("Press ESC to exit\n")

    # Initialize waste tracker
    # separation_threshold: pixel distance to consider hand and waste as separated (default: 50)
    # littering_time_threshold: seconds before marking waste as littered (default: 2.0)
    waste_tracker = WasteTracker(separation_threshold=80, littering_time_threshold=2.0)

    # Video input (change to 0 for webcam)
    cap = cv2.VideoCapture("photos/input4.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect hands and waste
        hand_results = detecthands(frame)
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

        cv2.imshow("Hand and Waste Detection with Littering Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()