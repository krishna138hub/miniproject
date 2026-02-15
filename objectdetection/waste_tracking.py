import time
import cv2
from collections import defaultdict

class WasteTracker:
    """
    Tracks waste objects and detects when they're thrown (separated from hands).
    Tags waste as littered if separation lasts longer than the threshold time.
    """

    def __init__(self, separation_threshold=50, littering_time_threshold=2.0):
        """
        Initialize waste tracker.

        Args:
            separation_threshold: Pixel distance threshold to consider hand and waste as separated
            littering_time_threshold: Time in seconds before marking waste as littered
        """
        self.separation_threshold = separation_threshold
        self.littering_time_threshold = littering_time_threshold

        # Track waste objects: {waste_id: {'bbox': (x1,y1,x2,y2), 'separated_time': float or None, 'littered': bool}}
        self.waste_objects = {}
        self.waste_counter = 0

    def get_distance(self, bbox1, bbox2):
        """
        Calculate minimum distance between two bounding boxes.

        Args:
            bbox1: (x1, y1, x2, y2) coordinates
            bbox2: (x1, y1, x2, y2) coordinates

        Returns:
            Minimum distance between the two boxes
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Center points
        cx1, cy1 = (x1_1 + x2_1) // 2, (y1_1 + y2_1) // 2
        cx2, cy2 = (x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2

        # Euclidean distance
        distance = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
        return distance

    def is_hand_waste_attached(self, hand_bbox, waste_bbox):
        """
        Check if hand and waste are attached (close together).

        Args:
            hand_bbox: Hand bounding box (x1, y1, x2, y2)
            waste_bbox: Waste bounding box (x1, y1, x2, y2)

        Returns:
            Boolean indicating if they're attached
        """
        distance = self.get_distance(hand_bbox, waste_bbox)
        return distance < self.separation_threshold

    def update(self, hand_results, waste_results, frame):
        """
        Update waste tracking based on current hand and waste detections.

        Args:
            hand_results: MediaPipe hand detection results
            waste_results: YOLO waste detection results
            frame: Current video frame

        Returns:
            Dictionary with littered waste information
        """
        current_time = time.time()

        # Extract hand bounding boxes
        hand_bboxes = []
        if hand_results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for handLms in hand_results.multi_hand_landmarks:
                xs = [int(lm.x * w) for lm in handLms.landmark]
                ys = [int(lm.y * h) for lm in handLms.landmark]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                hand_bboxes.append((x1, y1, x2, y2))

        # Extract waste bounding boxes
        waste_bboxes = []
        if waste_results:
            for result in waste_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    waste_bboxes.append((tuple(map(int, [x1, y1, x2, y2]))))

        littered_waste = []

        # Update existing waste objects
        waste_ids_to_remove = []
        for waste_id, waste_info in self.waste_objects.items():
            # Check if waste is still detected
            waste_still_detected = False
            for waste_bbox in waste_bboxes:
                if waste_info['bbox'] == waste_bbox:
                    waste_still_detected = True
                    break

            if not waste_still_detected:
                waste_ids_to_remove.append(waste_id)
                continue

            # Check if hand and waste are separated
            is_separated = True
            for hand_bbox in hand_bboxes:
                if self.is_hand_waste_attached(hand_bbox, waste_info['bbox']):
                    is_separated = False
                    break

            # Update separation status
            if is_separated:
                if waste_info['separated_time'] is None:
                    waste_info['separated_time'] = current_time

                # Check if waste should be marked as littered
                time_separated = current_time - waste_info['separated_time']
                if time_separated >= self.littering_time_threshold and not waste_info['littered']:
                    waste_info['littered'] = True
                    littered_waste.append({
                        'waste_id': waste_id,
                        'bbox': waste_info['bbox'],
                        'time_separated': time_separated
                    })
            else:
                # Hand and waste are attached, reset separation timer
                waste_info['separated_time'] = None
                waste_info['littered'] = False

        # Remove waste that's no longer detected
        for waste_id in waste_ids_to_remove:
            del self.waste_objects[waste_id]

        # Add new waste objects
        for waste_bbox in waste_bboxes:
            if waste_bbox not in [w['bbox'] for w in self.waste_objects.values()]:
                self.waste_counter += 1
                self.waste_objects[self.waste_counter] = {
                    'bbox': waste_bbox,
                    'separated_time': None,
                    'littered': False
                }

        return {
            'littered_waste': littered_waste,
            'active_waste': self.waste_objects,
            'separation_threshold': self.separation_threshold,
            'littering_time_threshold': self.littering_time_threshold
        }

    def draw_waste_status(self, frame, tracking_data):
        """
        Draw waste status on frame (attached/separated/littered).

        Args:
            frame: Video frame
            tracking_data: Output from update() method

        Returns:
            Modified frame
        """
        active_waste = tracking_data['active_waste']
        littered_waste_list = tracking_data['littered_waste']
        current_time = time.time()

        # Draw active waste
        for waste_id, waste_info in active_waste.items():
            x1, y1, x2, y2 = waste_info['bbox']

            if waste_info['littered']:
                # Draw in red for littered
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                status = "LITTERED"
                color = (0, 0, 255)
            elif waste_info['separated_time'] is not None:
                # Draw in yellow for separated
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                time_sep = current_time - waste_info['separated_time']
                status = f"SEPARATED ({time_sep:.1f}s)"
                color = (0, 255, 255)
            else:
                # Draw in green for attached
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                status = "ATTACHED"
                color = (0, 255, 0)

            label = f"Waste#{waste_id}: {status}"
            cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

