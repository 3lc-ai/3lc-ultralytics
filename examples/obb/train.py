from tlc_ultralytics import YOLO, Settings

settings = Settings(
    run_name="train-yolo11n-obb-dota8",
    project_name="3lc-yolo-examples",
    image_embeddings_dim=2,
    image_embeddings_reducer="umap",
    collect_loss=True,
)

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11n-obb.pt")

    # Train the model (this example is for demonstration purposes only - the
    # model we are trianing has already been trained on DOTA8)
    results = model.train(
        data="dota8.yaml",
        imgsz=640,
        workers=0,
        epochs=10,
        settings=settings,
    )
