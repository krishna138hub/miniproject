import cv2
import os

from objectdetection.hands import detecthands
from objectdetection.face import detectface
from objectdetection.wasteupdated import detectwaste
from objectdetection.wastetrackingupdated import WasteTracker
from objectdetection.roiupdated import detectroi, is_inside_roi


def main():
    print("Littering Detection System")
    print("Press ESC to exit\n")

    waste_tracker = WasteTracker(
        separation_threshold=80,
        littering_time_threshold=5.0
    )

    url = "http://192.168.0.6:8080/video"
    cap = cv2.VideoCapture(url)

    os.makedirs("littered_frames", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------- ROI Detection ----------------
        roi_result = detectroi(frame)   # Draws ROI
        roi_coords = roi_result.box     # (x1, y1, x2, y2)

        # ---------------- Hand Detection ----------------
        hand_results = detecthands(frame)

        # ---------------- Face Detection ----------------
        detectface(frame)   # Make sure your face.py already limits to 1 face

        # ---------------- Waste Detection ----------------
        waste_detections = detectwaste(frame)

        # Remove waste detections inside ROI (safe bin area)
        filtered_waste = []

        for waste in waste_detections:
            x1, y1, x2, y2 = waste["bbox"]

            if not is_inside_roi((x1, y1, x2, y2), roi_coords):
                filtered_waste.append(waste)

        # ---------------- Tracking ----------------
        littered_now = waste_tracker.update(
            hand_results,
            filtered_waste,
            frame
        )

        frame = waste_tracker.draw(frame)

        # ---------------- Save Evidence ----------------
        for item in littered_now:
            print(f"\n⚠️ LITTERING DETECTED - Waste ID {item['id']}")
            filename = f"littered_frames/litter_{item['id']}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved evidence: {filename}")

        cv2.imshow("Littering Detection System", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



