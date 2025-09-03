from tlc_ultralytics import YOLO

tables = {
    "val": "C:/Project/tlc-monorepo/tests/test_data/projects/GEOMETRY/datasets/default-dataset/tables/dota-8-val",
    "train": "C:/Project/tlc-monorepo/tests/test_data/projects/GEOMETRY/datasets/default-dataset/tables/dota-8-train",
}

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)

    # Train the model
    # results = model.collect(data="dota8.yaml", splits=("train",), imgsz=640, workers=0)
    results = model.collect(tables=tables, imgsz=640, workers=0)
