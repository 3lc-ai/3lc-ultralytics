from tlc_ultralytics import YOLO, Settings

if __name__ == "__main__":
    model = YOLO("yolo11n-cls.pt")

    settings = Settings(
        collection_epoch_start=1, # Collect metrics from the first epoch onwards
        collection_epoch_interval=1, # Collect metrics every epoch
        image_embeddings_dim=3, # Collect image embeddings and reduce to 3D
    )

    model.train(
        data="mnist",
        epochs=5,
        settings=settings,
    )
