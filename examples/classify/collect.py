from tlc_ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n-cls.pt")

    # Run inference on the ImageNet validation set with the yolo11n-cls weights
    model.collect(data="imagenet", splits=("val",))
