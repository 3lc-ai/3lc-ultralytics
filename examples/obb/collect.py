from tlc_ultralytics import YOLO, Settings

settings = Settings(
    run_name="yolo-obb-collection",
    project_name="3lc-yolo-examples",
    collect_loss=True,
    image_embeddings_reducer="umap",
    image_embeddings_dim=2,
)

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)

    # Collect metrics
    results = model.collect(
        data="dota8.yaml",
        splits=("train",),
        settings=settings,
        imgsz=640,
        workers=0,
    )
