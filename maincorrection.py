import cv2
import os

from objectdetection.hands import detecthands
from objectdetection.face import detectface
from objectdetection.wasteupdated import detectwaste
from objectdetection.wastetrackingupdated import WasteTracker


def main():
    print("Littering Detection System")
    print("Press ESC to exit\n")

    waste_tracker = WasteTracker(
        separation_threshold=80,
        littering_time_threshold=5.0
    )

    url = "photos/test.mp4"
    cap = cv2.VideoCapture(url)

    os.makedirs("objectdetection/littered_frames", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------- Hand Detection ----------------
        hand_results = detecthands(frame)

        # ---------------- Face Detection ----------------
        detectface(frame)

        # ---------------- Waste Detection (Filtered) ----------------
        waste_detections = detectwaste(frame)

        # ---------------- Tracking ----------------
        littered_now = waste_tracker.update(
            hand_results,
            waste_detections,
            frame
        )

        frame = waste_tracker.draw(frame)

        # ---------------- Save Evidence ----------------
        for item in littered_now:
            print(f"\n⚠️ LITTERING DETECTED - Waste ID {item['id']}")
            filename = f"littered_frames/litter_{item['id']}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved evidence: {filename}")

        # ---------------- Resize Display ----------------
        height, width = frame.shape[:2]
        scale = 0.7
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
