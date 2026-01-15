from tlc_ultralytics import create_tables_from_yaml_file

tables = create_tables_from_yaml_file("medical-pills.yaml", task="detect")
