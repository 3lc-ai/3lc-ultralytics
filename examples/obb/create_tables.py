from tlc_ultralytics import create_tables_from_yaml_file

tables = create_tables_from_yaml_file(
    dataset="dota8.yaml",
    task="obb",
)
