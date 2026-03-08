import cv2

class ROIResults:
    def __init__(self, box):
        self.box = box   # (x1, y1, x2, y2)


def detectroi(frame, width_ratio=0.3, height_ratio=0.3):
    h, w, _ = frame.shape

    # Define bottom-right ROI
    roi_width = int(w * width_ratio)
    roi_height = int(h * height_ratio)

    x1 = w - roi_width
    y1 = h - roi_height
    x2 = w
    y2 = h

    # Draw ROI box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(frame, "ROI", (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Return a simple result object like MediaPipe style
    return ROIResults((x1, y1, x2, y2))
