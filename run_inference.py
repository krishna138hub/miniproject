from ultralytics import YOLO
# =============================
# OPTIONAL INFERENCE
# =============================
def run_inference(image_path):
    model_path = "best.pt"
    model = YOLO(model_path)

    results = model(image_path)
    for result in results:
        print (result)
        result.show()
        result.save()

# =============================
# MAIN
# =============================
if __name__ == "__main__":

    run_inference("photos/roitest.jpeg")