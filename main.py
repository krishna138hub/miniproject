import cv2
import os         
import time

from components.waste import detectwaste
from components.wastetracking import WasteTracker
from components.hands import detecthands
from components.face import detectface
from components.bin import draw_bin, is_inside_roi
from constants.constants import IP_CAMERA_URL, PEOPLE_IN_FRAME_FOLDER, SAVE_FOLDER_LITTERED, VIDEO_FILE_PATH, VIDEO_SOURCE


def main():

    print("Littering Detection System")
    print("Press ESC to exit\n")

    cap = getVideo()

    littered_images =  SAVE_FOLDER_LITTERED
    people_folder = PEOPLE_IN_FRAME_FOLDER
    os.makedirs(littered_images, exist_ok=True) 
    os.makedirs(people_folder, exist_ok=True) 

    waste_tracker = WasteTracker(
        separation_threshold=80,
        littering_time_threshold=5.0
    )

    k_th_time = 0
    time_interval_in_milliseconds = 200

    screenshot_taken = False   # >>> ADDED (for 3rd second screenshot)

    increment_id = False

    while True:

        cap.set(cv2.CAP_PROP_POS_MSEC, k_th_time * time_interval_in_milliseconds)
        ret, frame = cap.read()

        if not ret:
            break

        # BIN ROI
        bin_box = draw_bin(frame)

        # Face detection
        face_boxes = detectface(frame)

        # Hand detection
        hand_results = detecthands(frame)

        # Waste detection
        waste_boxes = detectwaste(frame, increment_id)
        increment_id = False

        # Ignore waste inside bin
        waste_detections = []

        for waste in waste_boxes:
            if not is_inside_roi(waste["bbox"], bin_box.box):
                waste_detections.append(waste)

        littered_now = waste_tracker.update(
            hand_results,
            waste_detections,
            frame
        )
        # if littered_now:
        #     increment_id = True

        frame = waste_tracker.draw(frame)

        # >>> ADDED (3rd second screenshot)
        if face_boxes and not screenshot_taken:

            filename = f"{people_folder}/records.jpg"

            cv2.imwrite(filename, frame)

            print("Screenshot captured at 3rd second : ", filename)

            screenshot_taken = True
        # >>> END ADDED

        # >>> ADDED (litter detection screenshot)
        if littered_now:

            for obj in littered_now:

                print("LITTERING DETECTED")

                filename = f"{littered_images}/litter_{obj['id']}_{int(time.time())}.jpg"

                cv2.imwrite(filename, frame)

                print("Screenshot saved:", filename)

                #send email with the screenshot and details of the littering event (optional)
                # send face records to backend
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

        k_th_time = k_th_time + 1

    cap.release()
    cv2.destroyAllWindows()

def getVideo():
    switcher = {
        "WEB_CAMERA": getVideofromCamera,
        "VIDEO_FILE": getVideofromFile,
        "IP_CAMERA": getVideofromIpCamera
    }
    func = switcher.get(VIDEO_SOURCE, getVideofromCamera)
    return func()
    

def getVideofromCamera():
    cap = cv2.VideoCapture(0)
    return cap

def getVideofromIpCamera():
    cap = cv2.VideoCapture(IP_CAMERA_URL)
    return cap

def getVideofromFile():
    video_path = VIDEO_FILE_PATH
    cap = cv2.VideoCapture(video_path)
    return cap

if __name__ == "__main__":
    main()


