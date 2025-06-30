from tlc_ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n-seg.pt")

    model.collect(data="coco128-seg.yaml", splits=("val",))
