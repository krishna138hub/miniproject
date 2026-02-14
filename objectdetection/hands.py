import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_draw = mp.solutions.drawing_utils

def detecthands(frame):
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            # Get bounding box of hand
            xs = []
            ys = []
            for lm in handLms.landmark:
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))

            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, "Hand", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Draw landmarks
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)