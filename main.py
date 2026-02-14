import cv2
from objectdetection.hands import detecthands
from objectdetection.waste import detectwaste


def main():
    print("Application to detect hand and waste")
    # Video input (change to 0 for webcam)
    cap = cv2.VideoCapture("photos/input4.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detecthands(frame)
        detectwaste(frame)

        cv2.imshow("Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()