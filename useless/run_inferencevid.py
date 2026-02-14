from ultralytics import YOLO

# =============================
# OPTIONAL INFERENCE (VIDEO)
# =============================
def run_inference(video_path):
    model_path = "best.pt"
    model = YOLO(model_path)

    # stream=True gives frame-by-frame results (better for video)
    results = model(video_path, stream=True)

    for result in results:
        print(result)
        result.show()   # shows each frame
        result.save()   # saves output video in runs/detect/

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    run_inference("waste_video.mp4")
