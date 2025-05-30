from tlc_ultralytics import TLCYOLO

if __name__ == "__main__":
    model = TLCYOLO("yolo11n.pt")

    model.train(
        data="coco128.yaml",
        epochs=1,
        imgsz=160,
        workers=0,
    )
