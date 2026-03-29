import time
import cv2
from collections import deque


class WasteTracker:

    def __init__(self, separation_threshold=80, littering_time_threshold=5.0):
        self.separation_threshold = separation_threshold
        self.littering_time_threshold = littering_time_threshold
        self.waste_objects = {}
        self.littered_objects = set()
        self.next_id = 0

    def get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def distance(self, c1, c2):
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    def update(self, hand_results, waste_detections, frame):

        current_time = time.time()
        h, w, _ = frame.shape
        ground_line = int(h * 0.85)

        hand_bboxes = []

        if hand_results and hand_results.multi_hand_landmarks:
            for handLms in hand_results.multi_hand_landmarks:
                xs = [int(lm.x * w) for lm in handLms.landmark]
                ys = [int(lm.y * h) for lm in handLms.landmark]
                hand_bboxes.append((min(xs), min(ys), max(xs), max(ys)))

        new_littered = []

        if not waste_detections:
            return []

        updated_ids = set()

        for waste in waste_detections:
            bbox = waste["bbox"]
            center = self.get_center(bbox)

            # Find closest existing object
            min_dist = float('inf')
            closest_id = None
            for obj_id, obj in self.waste_objects.items():
                if obj["centroid_history"]:
                    history = list(obj["centroid_history"])
                    weights = [0.1 * (i + 1) for i in range(len(history))]
                    total_weight = sum(weights)
                    avg_x = sum(c[0] * w for c, w in zip(history, weights)) / total_weight
                    avg_y = sum(c[1] * w for c, w in zip(history, weights)) / total_weight
                    dist = self.distance(center, (avg_x, avg_y))
                    if dist < min_dist:
                        min_dist = dist
                        closest_id = obj_id

            if min_dist < self.separation_threshold and closest_id is not None:
                track_id = closest_id
            else:
                track_id = self.next_id
                self.next_id += 1
                self.waste_objects[track_id] = {
                    "centroid_history": deque(maxlen=10),
                    "bbox": bbox,
                    "state": "UNKNOWN",
                    "ground_time": None,
                    "littered": False,
                    "miss_count": 0
                }

            updated_ids.add(track_id)

            obj = self.waste_objects[track_id]
            obj["centroid_history"].append(center)

            # Check if previously littered and now above ground
            if obj["littered"] and center[1] < ground_line:
                obj["littered"] = False
                self.littered_objects.discard(track_id)

            attached = False
            for hand_bbox in hand_bboxes:
                if self.distance(center, self.get_center(hand_bbox)) < self.separation_threshold:
                    attached = True
                    break

            prev_center = obj["centroid_history"][-2] if len(obj["centroid_history"]) > 1 else center
            downward_motion = center[1] - prev_center[1] > 10

            if attached:
                obj["state"] = "HELD"
                obj["ground_time"] = None
                obj["littered"] = False
                if track_id in self.littered_objects:
                    self.littered_objects.discard(track_id)
            else:
                if obj["state"] == "HELD" and downward_motion:
                    obj["state"] = "DROPPED"

                if center[1] > ground_line:
                    if obj["ground_time"] is None:
                        obj["ground_time"] = current_time
                    if obj["littered"]:
                        obj["state"] = "LITTERED"
                    else:
                        obj["state"] = "ON_GROUND"
                    print(f"Object ID {track_id} on ground at time {obj['ground_time']}")

                if obj["ground_time"]:
                    time_on_ground = current_time - obj["ground_time"]
                    if time_on_ground >= self.littering_time_threshold and not obj["littered"]:
                        obj["state"] = "LITTERED"
                        obj["littered"] = True
                        self.littered_objects.add(track_id)
                        new_littered.append({"id": track_id, "bbox": bbox})

            obj["bbox"] = bbox

        # Remove objects not detected for 10 consecutive frames
        for obj_id in list(self.waste_objects.keys()):
            if obj_id not in updated_ids:
                self.waste_objects[obj_id]["miss_count"] += 1
                if self.waste_objects[obj_id]["miss_count"] >= 10:
                    if obj_id in self.littered_objects:
                        self.littered_objects.discard(obj_id)
                    del self.waste_objects[obj_id]
            else:
                self.waste_objects[obj_id]["miss_count"] = 0

        return new_littered

    def draw(self, frame):

        h, w, _ = frame.shape
        ground_line = int(h * 0.85)

        cv2.line(frame, (0, ground_line), (w, ground_line), (255, 255, 0), 2)

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


