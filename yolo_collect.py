import tlc
from torchvision.transforms import Compose, Lambda, Resize, ToTensor

from tlc_ultralytics import YOLO


# image_folder = "C:/Data/football/images"
def func(x):
    return x if x.shape[0] == 3 else x.repeat(3, 1, 1)


transforms = Compose([ToTensor(), Resize((640, 640)), Lambda(func)])
model = YOLO("yolo11n-cls.pt", verbose=False)
run = tlc.init("YOLO_EMBEDDINGS", "COCO_EMBEDDINGS_TRAIN")
image_folder = "C:/Data/coco/train2017"


if __name__ == "__main__":
    table = tlc.Table.from_image_folder(  # Compare with yolo_detect
        image_folder,
        include_label_column=False,
        project_name="YOLO_EMBEDDINGS",
        table_name="coco_train",
    )
    table.map(transforms)
    predictor = tlc.Predictor(
        model,
        layers=[151],
        preprocess_fn=lambda x: {"source": x, "verbose": False},
        unpack_dicts=True,
    )
    collector = tlc.EmbeddingsMetricsCollector(layers=[151])
    tlc.collect_metrics(
        table,
        collector,
        predictor,
        dataloader_args={"batch_size": 4, "num_workers": 4},
        split="MY_STREAM",
    )
    run.reduce_embeddings_by_foreign_table_url(table.url, method="pacmap")

    # import tlc
    # run = tlc.Run.from_url("C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/YOLO_EMBEDDINGS/runs/inflammable-lepton")
    # foreign_table_url = tlc.Table.from_url("C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/YOLO_EMBEDDINGS/datasets/default-dataset/tables/initial")
    print("===========COMPLETED===========")
    print(table.url)
    for t in run.metrics_tables:
        print(t.columns)
        print(t.get_foreign_table_url().to_absolute(t.url))
