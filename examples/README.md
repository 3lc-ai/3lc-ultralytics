# 3LC YOLO Integration Examples

This directory contains examples for using the 3LC YOLO integration for different computer vision tasks. For the main documentation, see the [repository README](../README.md).

## Register your dataset - Creating `tlc.Table`s

The first step when working with 3LC is to register your dataset as a 3LC Table. This creates a structured representation of your data that 3LC can track, analyze, and help you improve. Typically, a `tlc.Table` is created for each split of your dataset. The way to do this is different for each task, and we will show these here. While there are several ways to create `tlc.Table`s (including using a `tlc.TableWriter`), the most straightforward approach is to use the built-in convenience methods designed for common dataset formats. We will show the ones that are natively used in Ultralytics YOLO here.

### Classification

The dataset format used in YOLO for classification is the `ImageFolder` structure available in `torchvision`. This organizes images into class-specific subdirectories, making it easy to create 3LC Tables. To create a `tlc.Table` for an ImageFolder dataset, first arrange your images into separate directories for which class they belong to (see [`torchvision.ImageFolder`](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)), and then use `tlc.Table.from_image_folder`.

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

The most straightforward way to create `tlc.Table`s for this integration is using `tlc.Table.from_yolo`, which consumes a YOLO format dataset, or `tlc.Table.from_coco`, which consumes a COCO format dataset.

For YOLO datasets, specify the path to your dataset YAML file, indicate which split to load into the `tlc.Table`, and optionally provide a `datasets_dir`, which is prepended to relative paths in the YOLO Dataset YAML file:

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

For COCO format datasets, specify a path to the annotations JSON file and optionally a path to the images if the paths in the annotations file are relative.

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

For instance segmentation, you can use the same methods as object detection, with a few key modifications to handle segmentation data properly.

For YOLO datasets, set `task="segment"` to inform 3LC that your label files contain segmentation polygons rather than bounding boxes.

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

For COCO format datasets, set `task="segment"` and `segmentation_format="polygons"` since Ultralytics YOLO expects polygon data as input.

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

> NOTE: 3LC always stores instance segmentations as bitmasks internally. The polygon representation is computed from the masks only when the `tlc.Table` is accessed in Python.
> There will therefore often be a difference between the polygons returned by a `tlc.Table` and those in the input dataset, both in terms of the vertex locations and number of vertices.
> In practice, this leads to a small, but negligible, change in training results.

## Training

To run training with the 3LC integration, follow this pattern: instantiate the `YOLO` class with your chosen weights, then call `.train()` on the model, passing the tables to use and any additional arguments which are forwarded to Ultralytics YOLO:

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

This automatically creates a `tlc.Run` that you can visualize and analyze in the 3LC Dashboard.

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

Below you'll find examples organized by task, showing both training workflows and metrics collection:

### Classification Examples

- **[train.py](classify/train.py)**: Complete classification training workflow using the MNIST dataset. Perfect for understanding how classification differs from detection tasks.
- **[collect.py](classify/collect.py)**: Metrics collection and analysis on the ImageNet dataset. Great for evaluating large-scale classification models.

### Object Detection Examples

- **[train.py](detect/train.py)**: Complete object detection training workflow using a small Signatures dataset. Perfect for getting started with detection tasks.
- **[collect.py](detect/collect.py)**: Metrics collection and analysis for object detection on COCO128. Great for evaluating existing models or analyzing new datasets.

### Segmentation Examples

- **[train.py](segment/train.py)**: Complete instance segmentation workflow on the Carparts dataset. Demonstrates working with polygon annotations and mask outputs.
- **[collect.py](segment/collect.py)**: Metrics collection and analysis for instance segmentation on COCO128-seg. Shows segmentation-specific metrics and visualizations.

## Using the Examples

1. **Install the integration**: Follow the installation instructions in the [main README](../README.md) to set up the required dependencies
2. **Run the examples**: Execute the example files to see the integration in action:

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

Once you've run the examples or trained on your own data, you can begin the iterative improvement process:

1. **View results in the 3LC Dashboard**: Open the generated runs to explore detailed metrics, visualizations, and insights about your data and model performance.
2. **Modify your data**: Use the Dashboard to identify and fix issues in your dataset, creating new edited `Table`s with corrections and improvements.
3. **Retrain with improved data**: Use your edited `Table`s in new training runs to achieve better model performance and address the issues you identified.

For more advanced usage and configuration options, refer to the [main README](../README.md) and the [3LC Settings section](../README.md#3lc-settings).
