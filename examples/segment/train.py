from tlc_ultralytics import YOLO, Settings

if __name__ == "__main__":
    model = YOLO("yolo11s-seg.pt")

    settings = Settings(image_embeddings_dim=3)

    model.train(
        data="carparts-seg.yaml",
        epochs=10,
        settings=settings,
    )
