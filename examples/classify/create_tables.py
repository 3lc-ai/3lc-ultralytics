import tlc

train_table = tlc.Table.from_image_folder(
    root="/path/to/classes/train/",
    project_name="my_project_name",
    dataset_name="train",
    table_name="initial",
)
