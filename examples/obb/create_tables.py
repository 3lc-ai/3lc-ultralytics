import tlc

train_table = tlc.Table.from_yolo(
    dataset_yaml_file="/path/to/dataset.yaml",
    split="train",
    project_name="my_obb_project_name",
    dataset_name="train",
    table_name="initial",
    task="obb",
)
