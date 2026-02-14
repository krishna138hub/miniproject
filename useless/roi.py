import cv2

# Load image
img = cv2.imread("photos/roitest.jpeg")   # <-- change to your image path

if img is None:
    print("Image not found")
    exit()

h, w, _ = img.shape

# ---------------------------
# Define ROI (Top Right Corner)
# ---------------------------
roi_width = int(w * 0.5)    # 30% of image width
roi_height = int(h * 0.5)   # 30% of image height

x1 = w - roi_width
y1 = 0
x2 = w
y2 = roi_height

# Crop ROI
roi = img[y1:y2, x1:x2]

# ---------------------------
# Draw Bounding Box
# ---------------------------
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

# ---------------------------
# Show Results
# ---------------------------
cv2.imshow("Original Image with ROI Box", img)
cv2.imshow("Top Right ROI", roi)

cv2.waitKey(0)
cv2.destroyAllWindows()
