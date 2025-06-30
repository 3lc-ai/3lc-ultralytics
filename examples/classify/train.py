from tlc_ultralytics import Settings, YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n-cls.pt")

    settings = Settings(image_embeddings_dim=3)

    model.train(
        data="mnist",
        epochs=10,
        settings=settings,
    )
