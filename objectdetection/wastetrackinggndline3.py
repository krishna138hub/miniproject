import time
import cv2


class WasteTracker:

    def __init__(self, separation_threshold=80, littering_time_threshold=5.0):
        self.separation_threshold = separation_threshold
        self.littering_time_threshold = littering_time_threshold
        self.waste_objects = {}

    def get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def distance(self, c1, c2):
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    def update(self, hand_results, waste_detections, frame):

        current_time = time.time()
        h, w, _ = frame.shape
        self.ground_line = int(h * 0.85)

        # ---- HAND EXTRACTION ----
        hand_bboxes = []
        if hand_results and hand_results.multi_hand_landmarks:
            for handLms in hand_results.multi_hand_landmarks:
                xs = [int(lm.x * w) for lm in handLms.landmark]
                ys = [int(lm.y * h) for lm in handLms.landmark]
                hand_bboxes.append((min(xs), min(ys), max(xs), max(ys)))

        littered_now = []

        if not waste_detections:
            return []

        waste = waste_detections[0]
        track_id = waste["id"]
        bbox = waste["bbox"]
        center = self.get_center(bbox)

        if track_id not in self.waste_objects:
            self.waste_objects = {}
            self.waste_objects[track_id] = {
                "bbox": bbox,
                "state": "UNKNOWN",
                "prev_center": center,
                "ground_time": None,
                "littered": False,
                "screenshot_taken": False
            }

        obj = self.waste_objects[track_id]
        prev_center = obj["prev_center"]

        # ----- Check if held -----
        attached = False
        for hand_bbox in hand_bboxes:
            if self.distance(center, self.get_center(hand_bbox)) < self.separation_threshold:
                attached = True
                break

        downward_motion = center[1] - prev_center[1] > 10

        if attached:
            obj["state"] = "HELD"
            obj["ground_time"] = None
            obj["littered"] = False

        else:

            if obj["state"] == "HELD" and downward_motion:
                obj["state"] = "DROPPED"

            if center[1] > self.ground_line:
                if obj["ground_time"] is None:
                    obj["ground_time"] = current_time
                obj["state"] = "ON_GROUND"

            if obj["ground_time"]:

                time_on_ground = current_time - obj["ground_time"]

                # ---- TAKE SCREENSHOT AT 3 SECONDS ----
                if time_on_ground >= 3 and not obj["screenshot_taken"]:
                    filename = f"littered_frames/litter_3sec_{track_id}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 3-second screenshot saved: {filename}")
                    obj["screenshot_taken"] = True

                # ---- FINAL LITTER DETECTION ----
                if time_on_ground >= self.littering_time_threshold and not obj["littered"]:
                    obj["state"] = "LITTERED"
                    obj["littered"] = True
                    littered_now.append({
                        "id": track_id,
                        "bbox": bbox,
                        "time": time_on_ground
                    })

        obj["bbox"] = bbox
        obj["prev_center"] = center

        return littered_now

    def draw(self, frame):

        h, w, _ = frame.shape

        # -------- DRAW GROUND LINE --------
        cv2.line(frame,
                 (0, self.ground_line),
                 (w, self.ground_line),
                 (255, 0, 0),
                 3)

        cv2.putText(frame,
                    "GROUND LINE",
                    (10, self.ground_line - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2)

        # -------- DRAW OBJECT STATES --------
        for track_id, obj in self.waste_objects.items():

            x1, y1, x2, y2 = obj["bbox"]

            if obj["state"] == "HELD":
                color = (0, 255, 0)
            elif obj["state"] == "DROPPED":
                color = (0, 255, 255)
            elif obj["state"] == "ON_GROUND":
                color = (255, 0, 255)
            elif obj["state"] == "LITTERED":
                color = (0, 0, 255)
            else:
                color = (200, 200, 200)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame,
                        f"ID {track_id}: {obj['state']}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

        return frame

