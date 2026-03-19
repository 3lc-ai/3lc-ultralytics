from tlc_ultralytics import create_tables_from_yaml_file

tables = create_tables_from_yaml_file(
    dataset="coco128.yaml",
    task="detect",
)
