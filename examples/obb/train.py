from tlc_ultralytics import YOLO, Settings

# tables = {
#     "val": "C:/Project/tlc-monorepo/tests/test_data/projects/GEOMETRY/datasets/default-dataset/tables/dota-8-val",
#     "train": "C:/Project/tlc-monorepo/tests/test_data/projects/GEOMETRY/datasets/default-dataset/tables/dota-8-train",
# }

data_yaml = "D:/Data/DOTAv1/data.yaml"
settings = Settings(
    run_name="train-yolo11n-obb",
    project_name="3lc-yolo-examples-debug",
)

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)

    # Train the model
    # results = model.collect(data="dota8.yaml", splits=("train",), imgsz=640, workers=0)
    # results = model.collect(tables=tables, imgsz=640, workers=0)
    # results = model.train(data="dota8.yaml", imgsz=640, workers=0, epochs=1, settings=settings)
    results = model.train(data=data_yaml, imgsz=640, workers=0, epochs=9, settings=settings)
