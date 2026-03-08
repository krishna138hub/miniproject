import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection

face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6
)


def detectface(frame):

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)

    if results.detections:

        for det in results.detections:

            box = det.location_data.relative_bounding_box

            x1 = int(box.xmin * w)
            y1 = int(box.ymin * h)
            x2 = int((box.xmin + box.width) * w)
            y2 = int((box.ymin + box.height) * h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)

            cv2.putText(frame,
                        "Face",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 150, 0),
                        2)

    return results
