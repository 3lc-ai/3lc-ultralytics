from tlc_ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26n-sem.pt")

    model.collect(data="cityscapes8.yaml", splits=("train", "val"))
