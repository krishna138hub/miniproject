import cv2
import os            # >>> ADDED
import time          # >>> ADDED

from waste1 import detectwaste
from wastetracking1 import WasteTracker
from hands1 import detecthands
from face1 import detectface
from roi1 import detectroi, is_inside_roi


def main():

    print("Littering Detection System")
    print("Press ESC to exit\n")

    video_path = "photos2/good2.mp4"
    cap = cv2.VideoCapture(video_path)

    save_folder = "littered_frames1"
    save_folder1 = "people_in_frame"# >>> ADDED
    os.makedirs(save_folder, exist_ok=True) # >>> ADDED

    waste_tracker = WasteTracker(
        separation_threshold=80,
        littering_time_threshold=5.0
    )

    second = 0

    screenshot_taken = False   # >>> ADDED (for 3rd second screenshot)

    while True:

        cap.set(cv2.CAP_PROP_POS_MSEC, second * 200)
        ret, frame = cap.read()

        if not ret:
            break

        # BIN ROI
        roi_results = detectroi(frame)

        # Face detection
        detectface(frame)

        # Hand detection
        hand_results = detecthands(frame)

        # Waste detection
        waste_boxes = detectwaste(frame)

        # Ignore waste inside bin
        filtered_waste = []

        for waste in waste_boxes:
            if not is_inside_roi(waste["bbox"], roi_results.box):
                filtered_waste.append(waste)

        littered_now = waste_tracker.update(
            hand_results,
            filtered_waste,
            frame
        )

        frame = waste_tracker.draw(frame)

        # >>> ADDED (3rd second screenshot)
        if second == 3 and not screenshot_taken:

            filename = f"{save_folder1}/records.jpg"

            cv2.imwrite(filename, frame)

            print("Screenshot captured at 3rd second")

            screenshot_taken = True
        # >>> END ADDED

        # >>> ADDED (litter detection screenshot)
        if littered_now:

            for obj in littered_now:

                print("LITTERING DETECTED")

                filename = f"{save_folder}/litter_{obj['id']}_{int(time.time())}.jpg"

                cv2.imwrite(filename, frame)

                print("Screenshot saved:", filename)
        else:
            print("No littering detected at this moment.")

        # >>> END ADDED

        # Resize screen for visualization
        screen_w = 1280
        screen_h = 720

        h, w = frame.shape[:2]

        if w > screen_w or h > screen_h:
            scale = min(screen_w / w, screen_h / h)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        cv2.imshow("Littering Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        second = second + 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


