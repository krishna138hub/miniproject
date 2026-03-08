import cv2


class ROIResults:
    def __init__(self, box):
        self.box = box  # (x1, y1, x2, y2)


def detectroi(frame):
    h, w, _ = frame.shape

    # ---- Adjusted ROI to match bin position in your video ----
    x1 = int(w * 0.679)
    x2 = int(w * 0.825)

    y1 = int(h * 0.315)
    y2 = int(h * 0.585)

    # Draw ROI box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(frame, "BIN",
                (x1 + 10, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2)

    return ROIResults((x1, y1, x2, y2))

def is_inside_roi(bbox, roi_box):
    bx1, by1, bx2, by2 = bbox
    rx1, ry1, rx2, ry2 = roi_box

    # Check if waste box center is inside ROI
    center_x = (bx1 + bx2) // 2
    center_y = (by1 + by2) // 2

    return (rx1 <= center_x <= rx2) and (ry1 <= center_y <= ry2)
