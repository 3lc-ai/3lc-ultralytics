from tlc_ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")

    model.train(
        data="coco8.yaml",
        epochs=1,
        batch=4,
        imgsz=64,
        device="cpu",
        workers=0,
        seed=42,
        deterministic=True,
        save=False,
        plots=False,
        project="tests/tmp/tlc_ultralytics",
        name="train_detect",
    )
