# 3LC YOLO Integration Examples

This directory contains examples for using the 3LC YOLO integration for different computer vision tasks. For the main documentation, see the [repository README](../README.md).

## Register your dataset - Creating `tlc.Table`s

The first step when working with 3LC is to register your dataset as a 3LC Table. Typically, a `tlc.Table` is created for each split of your dataset. The way to do this is different for each task, and we will show these here. There are several ways of creating `tlc.Table`s, such as using a `tlc.TableWriter`, but most of the time the easiest is to use convenience methods that are available for common dataset formats. We will show the ones that are natively used in Ultralytics YOLO here.

### Classification

The dataset format used in YOLO for classification is the `ImageFolder` available in `torchvision`. Check out ... for more details. To create a `tlc.Table` for an ImageFolder dataset, first arrange your images into separate directories for which class they belong to (see torchvision ImageFolder API docs), and then use `tlc.Table.from_image_folder`.

```python
import tlc

root = "/path/to/classes/train/"

train_table = tlc.Table.from_image_folder(
    root=root,
    project_name="my_project_name",
    dataset_name="train",
    table_name="initial",
)
```

### Object Detection

The easiest way to create `tlc.Table`s to use in the integration is by means of `tlc.Table.from_yolo`, which consumes a YOLO format dataset, or `tlc.Table.from_coco`, which consumes a COCO format dataset.

For YOLO datasets, provide the path to the dataset YAML file, which split to read into the `tlc.Table`, and optionally a `datasets_dir`, which is prepended to relative paths in the YOLO Dataset YAML file:

```python
import tlc

train_table = tlc.Table.from_yolo(
    dataset_yaml_file="/path/to/dataset.yaml",
    split="train",
    project_name="my_detection_project_name",
    dataset_name="train",
    table_name="initial",
    task="detect",
)
```

For COCO format datasets, provide a path to the annotations json file and optionally a path to the images if the paths in the annotations file are relative.

```python
import tlc

train_table = tlc.Table.from_coco(
    annotations_file="/path/to/annotations.json",
    image_folder="/path/to/image_folder/",
    project_name="my_detection_project_name",
    dataset_name="train",
    table_name="initial",
    task="detect",
)
```

### Instance Segmentation

For instance segmentation, the same methods should be used as for Object Detection, with a few slight modifications.

For YOLO datasets, provide `task="segment"` to tell 3LC that the label files contain segmentation polygons instead of bounding boxes.

```python
import tlc

train_table = tlc.Table.from_yolo(
    dataset_yaml_file="/path/to/dataset.yaml",
    split="train",
    project_name="my_detection_project_name",
    dataset_name="train",
    table_name="initial",
    task="segment",
)
```

For COCO format datasets, provide `task="segment"` and set `segmentation_format="polygons"` as Ultralytics YOLO expects polygon data as input.

```python
import tlc

train_table = tlc.Table.from_coco(
    annotations_file="/path/to/annotations.json",
    image_folder="/path/to/image_folder/",
    project_name="my_detection_project_name",
    dataset_name="train",
    table_name="initial",
    task="segment",
    segmentation_format="polygons",
)
```

> NOTE: 3LC always stores instance segmentations as bitmasks internally. Only when the `tlc.Table` is accessed in Python, the polygon representation is computed from the masks.
> There will therefore often be a difference between the polygons returned by a `tlc.Table` and those in the input dataset, both in terms of the vertex locations and number of vertices.
> In practice, this leads to a small, but negligible, change in training results.

## Training

To run training with the 3LC integration, the main pattern is to instantiate the main `YOLO` class with a set of weights, and then to call `.train()` on this model, passing the tables to use and any additional arguments which are forwarded to Ultralytics YOLO:

```python
from tlc_ultralytics import YOLO

model = YOLO("<path to weights>.pt")

model.train(
    tables={"train": my_train_table, "val": my_val_table},
    epochs=10,
    imgsz=640,
    ...
)
```

This creates a `tlc.Run` which can be visualized in the 3LC Dashboard.

## Collection

To only run inference with a set of trained weights, the `.collect()` method can be used instead, passing one or more `tlc.Table`s to run validation on. This will run `.val()` on each `tlc.Table` and reduce any collected metrics at the end. Additional kwargs to `.collect()` are forwarded to `.val()`.

```python
from tlc_ultralytics import YOLO

model = YOLO("<path to weights>.pt")

model.collect(
    tables={"train": my_train_table, "val": my_val_table},
    imgsz=640,
    ...
)
```

## Examples

See the following sections for examples of training and collection, grouped by task:

### Classification Examples

- **[train.py](classify/train.py)**: Basic classification training example using the MNIST dataset.
- **[collect.py](classify/collect.py)**: Metrics collection on the Imagenet dataset.

### Object Detection Examples

- **[train.py](detect/train.py)**: Basic object detection training example using a small Signatures dataset.
- **[collect.py](detect/collect.py)**: Metrics collection example for object detection on COCO128.

### Segmentation Examples

- **[train.py](segment/train.py)**: Basic instance segmentation example on the Carparts dataset.
- **[collect.py](segment/collect.py)**: Metrics collection example for object detection on COCO128-seg.

## Using the Examples

1. **Install the integration**: Follow the installation instructions in the [main README](../README.md)
1. **Run the examples**: Execute the example files to see how to use the integration:

```bash
# Classification
python examples/classify/train.py
python examples/classify/collect.py

# Object Detection
python examples/detect/train.py
python examples/detect/collect.py

# Instance segmentation
python examples/segment/train.py
python examples/segment/collect.py
```

## Next Steps

After running the examples or training on your own data, you can iteratively:

1. **View results in the 3LC Dashboard**: Open the generated runs to explore metrics and visualizations.
1. **Modify your data**: Use the Dashboard to debug issues in the dataset, creating new edited `Table`s.
1. **Retrain with improved data**: Use the edited `Table`s in a new training run for better model performance.

For more advanced usage and configuration options, refer to the [main README](../README.md) and the [3LC Settings section](../README.md#3lc-settings).
