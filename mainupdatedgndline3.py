import cv2
import os

from objectdetection.hands import detecthands
from objectdetection.face import detectface
from objectdetection.wasteupdated import detectwaste
from objectdetection.wastetrackinggndline3 import WasteTracker
#from objectdetection.roiupdated import detectroi, is_inside_roi


def main():
    print("Littering Detection System")
    print("Press ESC to exit\n")

    waste_tracker = WasteTracker(
        separation_threshold=80,
        littering_time_threshold=5.0
    )
    url = "photos/bad2.mp4"
    cap = cv2.VideoCapture(url)

    os.makedirs("littered_frames", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------- ROI Detection ----------------
        #roi_result = detectroi(frame)
        #roi_coords = roi_result.box

        # ---------------- Hand Detection ----------------
        hand_results = detecthands(frame)

        # ---------------- Face Detection ----------------
        detectface(frame)

        # ---------------- Waste Detection ----------------
        waste_detections = detectwaste(frame)

        # Remove waste inside bin area
        filtered_waste = []

        #for waste in waste_detections:
           # x1, y1, x2, y2 = waste["bbox"]

            #if not is_inside_roi((x1, y1, x2, y2), roi_coords):
                #filtered_waste.append(waste)

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
