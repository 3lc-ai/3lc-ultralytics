from tlc_ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.collect(data="dota8.yaml", splits=("train",), imgsz=640, workers=0)
