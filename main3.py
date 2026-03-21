import cv2
import os
import time
import requests

from components.waste import detectwaste
from components.wastetracking1 import WasteTracker
from components.hands import detecthands
from components.face import detectface
from components.bin import draw_bin, is_inside_roi
from constants.constants import IP_CAMERA_URL, PEOPLE_IN_FRAME_FOLDER, SAVE_FOLDER_LITTERED, VIDEO_FILE_PATH, VIDEO_SOURCE


# ---------------- TELEGRAM SETTINGS ----------------
#BOT_TOKEN = 'blahblablah'   # Your bot token
#CHAT_ID = "123456789"                                       # Your chat ID


def send_to_telegram(image_path, caption):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    try:
        with open(image_path, "rb") as img:
            files = {"photo": img}
            data = {"chat_id": CHAT_ID, "caption": caption}

            response = requests.post(url, files=files, data=data)

            if response.status_code == 200:
                print("Image sent to Telegram successfully")
            else:
                print("Failed to send image:", response.text)
    except Exception as e:
        print("Telegram error:", e)


def main():
    print("Littering Detection System")
    print("Press ESC to exit\n")

    cap = getVideo()

    littered_images = SAVE_FOLDER_LITTERED
    people_folder = PEOPLE_IN_FRAME_FOLDER
    os.makedirs(littered_images, exist_ok=True)
    os.makedirs(people_folder, exist_ok=True)

    waste_tracker = WasteTracker(
        separation_threshold=80,
        littering_time_threshold=5.0
    )

    k_th_time = 0
    time_interval_in_milliseconds = 200

    screenshot_taken = False
    last_telegram_time = 0   # ✅ prevent spam (cooldown)

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

        frame = waste_tracker.draw(frame)

        # ---------------- FACE SCREENSHOT ----------------
        if face_boxes and not screenshot_taken:
            timestamp = int(time.time())
            face_filename = f"{people_folder}/person_{timestamp}.jpg"

            cv2.imwrite(face_filename, frame)
            print("Face screenshot saved:", face_filename)

            screenshot_taken = True

        # ---------------- LITTER DETECTION ----------------
        if littered_now:
            for obj in littered_now:
                print("LITTERING DETECTED")

                timestamp = int(time.time())
                litter_filename = f"{littered_images}/litter_{obj['id']}_{timestamp}.jpg"

                cv2.imwrite(litter_filename, frame)
                print("Litter screenshot saved:", litter_filename)

                # ✅ Telegram cooldown (10 sec)
                if time.time() - last_telegram_time > 10:
                    send_to_telegram(face_filename, "🚨 Littering detected!")
                    last_telegram_time = time.time()

        else:
            print("No littering detected at this moment.")

        # ---------------- DISPLAY ----------------
        screen_w = 1280
        screen_h = 720

        h, w = frame.shape[:2]
        if w > screen_w or h > screen_h:
            scale = min(screen_w / w, screen_h / h)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        cv2.imshow("Littering Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        k_th_time += 1

    cap.release()
    cv2.destroyAllWindows()


# ---------------- VIDEO SOURCE ----------------
def getVideo():
    switcher = {
        "WEB_CAMERA": getVideofromCamera,
        "VIDEO_FILE": getVideofromFile,
        "IP_CAMERA": getVideofromIpCamera
    }
    func = switcher.get(VIDEO_SOURCE, getVideofromCamera)
    return func()


def getVideofromCamera():
    return cv2.VideoCapture(0)


def getVideofromIpCamera():
    return cv2.VideoCapture(IP_CAMERA_URL)


def getVideofromFile():
    return cv2.VideoCapture(VIDEO_FILE_PATH)


if __name__ == "__main__":
    main()
