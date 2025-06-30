from tlc_ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n-cls.pt")

    model.collect(data="imagenet", splits=("val",))
