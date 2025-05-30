from tlc_ultralytics import TLCYOLO

if __name__ == "__main__":
    model = TLCYOLO("yolo11n.pt")

    model.collect(
        data="coco128.yaml",
        splits=["train", "val"],
        imgsz=160,
        batch=8,
    )
