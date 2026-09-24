from tlc_ultralytics import YOLO, Settings

if __name__ == "__main__":
    model = YOLO("yolo26n-sem.pt")

    settings = Settings(image_embeddings_dim=3, collect_loss=True)

    model.train(
        data="cityscapes8.yaml",
        epochs=10,
        settings=settings,
    )
