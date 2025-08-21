from tlc_ultralytics import Settings, YOLO

if __name__ == "__main__":
    model = YOLO("yolo11m.pt")

    settings = Settings(
        image_embeddings_dim=2,  # Collect image embeddings and reduce to 2D
    )

    model.train(
        data="signature.yaml",
        epochs=3,
        imgsz=160,
        workers=4,
        settings=settings,
    )
