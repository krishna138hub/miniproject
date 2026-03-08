from ultralytics import YOLO
# =============================
# OPTIONAL INFERENCE
# =============================
def run_inference(image_path):
    model_path = "best.pt"
    model = YOLO(model_path)
    print(model.names)

    results = model(image_path)
    for result in results:
        print (result)
        result.show()
        result.save()



# =============================
# MAIN
#hello
# =============================
if __name__ == "__main__":

    run_inference("photos1/waste.avif")