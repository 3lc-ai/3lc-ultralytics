from tlc_ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n-cls.pt")

    model.train(
        data="mnist",
        epochs=100,
        imgsz=32,
        workers=0,
    )
