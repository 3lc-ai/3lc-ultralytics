from tlc_ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11m.pt")

    model.train(
        data="signature.yaml",
        epochs=3,
        imgsz=160,
        workers=0,
    )
