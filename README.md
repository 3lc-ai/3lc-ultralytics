<p align="center">
<img src="https://3lc.ai/wp-content/uploads/2023/09/3LC-Logo_Footer.svg">
</p>

<h1 align="center">3LC YOLO Integration</h1>

<div align="center">

![PyPI](https://img.shields.io/pypi/v/3lc-ultralytics?logo=pypi&logoColor=white) [![Discord](https://img.shields.io/badge/discord-3LC-5865F2?logo=discord&logoColor=white)](https://discord.gg/fwnwFtfafC)

</div>

<p align="center">
<a href="#quick-start">Quick Start</a> •
<a href="#working-with-datasets">Working with Datasets</a> •
<a href="#task-specific-configuration">Task-Specific Configuration</a> •
<a href="#metrics-collection-only">Metrics Collection</a> •
<a href="#3lc-settings">3LC Settings</a> •
<a href="#frequently-asked-questions">FAQ</a>
</p>

<p align="center">
Ultralytics YOLO classification, object detection and segmentation with 3LC integrated.
</p>

## About 3LC

[3LC](https://3lc.ai) is a tool which enables data scientists to improve machine learning models in a data-centric fashion. It collects per-sample predictions and metrics, allows viewing and modifying the dataset in the context of those predictions in the 3LC Dashboard, and rerunning training with the revised dataset.

3LC is free for non-commercial use.

![3LC Dashboard Overview](https://github.com/3lc-ai/ultralytics/blob/tlc-integration/ultralytics/utils/tlc/_static/dashboard.png?raw=true)

## Quick Start

### Installation

Install the package and requirements into a virtual environment:

```bash
pip install 3lc-ultralytics
```

### Basic Training

Import `YOLO` from `tlc_ultralytics` and start training with 3LC integration:

```python
from tlc_ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="coco128.yaml", epochs=1)
```

In the background, 3LC creates `tlc.Table`s for each split and a `tlc.Run`, which can be opened in the 3LC Dashboard.

### Examples

Check out the [examples directory](examples/) for complete training and metrics collection examples for each supported task:

- **Classification**: [examples/classify/train.py](examples/classify/train.py) and [examples/classify/collect.py](examples/classify/collect.py)
- **Object Detection**: [examples/detect/train.py](examples/detect/train.py) and [examples/detect/collect.py](examples/detect/collect.py)
- **Segmentation**: [examples/detect/train.py](examples/segment/train.py) and [examples/detect/collect.py](examples/segment/collect.py)

## Working with Datasets

The integration supports three ways of providing the data use. These are listed below:

<details open>
<summary><strong>Using 3LC Tables Directly (Recommended)</strong></summary>

The recommended way of providing the data to use is to pass `tlc.Table`s or `tlc.Url`s to `tlc.Table`s directly.

To learn how to create `tlc.Table`s for your dataset, check out the [examples directory](/examples/).

```python
from tlc_ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Using table instances
tables = {
    "train": my_train_table, 
    "val": my_val_table
}
model.train(tables=tables)

# Using table URLs
tables = {
    "train": "/url/or/path/to/train/table",
    "val": "/url/or/path/to/val/table"
}
model.train(tables=tables, ...)
```

When `tables` is provided, any value of `data` is ignored. In training, the table for the key `"train"` is used for training, and `"val"` or `"test"` for validation (val takes precedence).

</details>

<details>
<summary>Using Existing YOLO Datasets</summary>

Another alternative is to pass the argument `data` like in vanilla Ultralytics, pointing to a YOLO dataset YAML file. See the [Ultralytics Documentation](https://docs.ultralytics.com/datasets/) to learn more.

3LC parses these datasets and creates `tlc.Table`s for each split, which can be viewed in the Dashboard. Once you make new versions of your data in the 3LC Dashboard, you can use the same command with `data=<path to your dataset>`, and the latest version will be used automatically. The integration will use default values for `project_name` (the name of the provided dataset yaml file, e.g. `"my_dataset"` for `"data=/path/to/my_dataset.yaml"`), `dataset_name` (the split, e.g. `"train"`) and `table_name` (`"initial"`).

</details>

<details>
<summary>Using 3LC YAML Files</summary>

The third and final way of providing the data is through what we call a 3LC YOLO YAML file. This is available in order to be compatible with the corresponding [YOLOv5 Integration](https://github.com/3lc-ai/yolov5).

Create a dataset YAML file, and provide the `tlc.Url` to each split in the file:

```yaml
# my_dataset.yaml
train: /path/to/train/table
val: /path/to/val/table

# With versioning
train: /path/to/train/table:latest
val: s3://path/to/val/table
```

Then use it with the `3LC://` prefix to specify that it is a 3LC YOLO YAML file:

```python
from tlc_ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="3LC://my_dataset.yaml")
```

Note: `names` and `nc` are not needed since the `tlc.Table`s themselves contain the category names and indices.

</details>

## Task-Specific Configuration

### Classification

For image classification, you can provide `image_column_name` and `label_column_name` when calling `model.train()`, `model.val()` and `model.collect()` if you are providing your own table which has different column names to those expected by 3LC.

### Object Detection

In addition to tables created with `Table.from_yolo()` (which is called internally when you provide a YOLO dataset), it is also possible to use tables with the COCO format used in the 3LC Detectron2 integration. If you have created 3LC tables in the Detectron2 integration, you can also use this `tlc.Table` in this integration!

### Segmentation

Working with instance segmentation in the 3LC integration is similar to object detection. If you are using `tlc.Table.from_yolo()` to create your tables, make sure to pass `task="segment"`, as the default is `"detect"`.

For instance segmentation, you can provide `image_column_name` and `label_column_name` when calling `model.train()`, `model.val()` and `model.collect()` if you are providing your own table which has different column names to those expected by 3LC. Note that the `label_column_name` should be the path to the segmentation label field within the table schema, with the default being `"segmentations.instance_properties.label"` which points to the category labels for each segmentation instance.

### Unsupported Tasks

Some YOLO tasks can not yet be visualized in the 3LC Dashboard, but these are on the roadmap and will be made available in the future:

- **Pose Estimation**: Not yet supported. Let us know on Discord if you would like this to be supported!
- **OBB (Oriented Object Detection)**: Not yet supported. Let us know on Discord if you would like this to be supported!

## Metrics Collection Only

It is possible to create runs where only metrics collection, and no training, is performed. This is useful when you already have a trained model and would like to collect metrics, or if you would like to collect metrics on a different dataset to the one you trained and validated on.

Use the method `model.collect()` to perform metrics collection only. Either pass `data` (a path to a yaml file) and `splits` (an iterable of split names to collect metrics for), or a dictionary `tables` like detailed in the previous section, to define which data to collect metrics on. This will create a run, collect the metrics on each split by calling `model.val()` and finally reduce any embeddings that were collected. Any additional arguments, such as `imgsz` and `batch`, are forwarded as `model.val(**kwargs)`.

### Example: Metrics Collection

```python
from tlc_ultralytics import Settings, YOLO

model = YOLO("yolo11m.pt")

settings = Settings(
    image_embeddings_dim=2,
    conf_thres=0.2,
)

model.collect(
    data="coco128.yaml",
    splits=("train", "val"),
    settings=settings,
    batch=32,
    imgsz=320
)
```

See [examples/detect/collect.py](examples/detect/collect.py) for a complete metrics collection example.

## 3LC Settings

The integration offers a rich set of settings and features which can be set through an instance of `Settings`, which are in addition to the regular YOLO settings. They allow specifying which metrics to collect, how often to collect them, and whether to use sampling weights during training.

The available 3LC settings can be seen in the `Settings` class in [settings.py](src/tlc_ultralytics/settings.py).

Providing invalid values (or combinations of values) will either log an appropriate warning or raise an error, depending on the case.

### Image Embeddings

Image embeddings can be collected by setting `image_embeddings_dim` to 2 or 3. Similar images, as seen by the model, tend to be close to each other in this space. In the 3LC Dashboard these embeddings can be visualized, allowing you to find similar images, duplicates and imbalances in your dataset, and take appropriate actions to mitigate these issues.

The way in which embeddings are collected is different for the different tasks:

- **Classification**: The integration scans your model for the first occurrence of a `torch.nn.Linear` layer. The inputs to this layer are used to extract image embeddings.
- **Object Detection and Instance Segmentation**: The output of the spatial pooling function is used to extract embeddings.

You can change which `3lc`-supported reducer to use by setting `image_embeddings_reducer`. `pacmap` is the default.

### Run Properties

Use `project_name`, `run_name` and `run_description` to customize the `tlc.Run` that is created. Any tables created by the integration will be under the `project_name` provided here. If these settings are not set, appropriate defaults are used instead.

### Sampling Weights

Use `sampling_weights=True` to enable the usage of sampling weights. This resamples the data presented to the model according to the weight column in the `Table`. If a sample has weight 2.0, it is twice as likely to appear as a particular sample with weight 1.0. Any given sample can occur multiple times in one epoch. This setting only applies to training.

### Exclude Zero Weight Samples

Use `exclude_zero_weight_training=True` (only applies to training) and `exclude_zero_weight_collection=True` to eliminate rows with weight 0.0. If your table has samples with weight 0.0, this will effectively reduce the size of the dataset (i.e. reduce the number of iterations per epoch).

### Metrics Collection Settings

- **`collection_val_only=True`**: Disable metrics collection on the training set. This only applies to training.
- **`collection_disable=True`**: Disable metrics collection entirely. This only applies to training. A run will still be created, and hyperparameters and aggregate metrics will be logged to 3LC.
- **`collection_epoch_start` and `collection_epoch_interval`**: Define when to collect metrics during training. The start epoch is 1-based, i.e. 1 means after the first epoch. As an example, `collection_epoch_start=1` with `collection_epoch_interval=2` means metrics collection will occur after the first epoch and then every other epoch after that.

## Dashboard Output

When viewing all your YOLO runs in the 3LC Dashboard, charts will show up with per-epoch aggregate metrics produced by YOLO for each run. This allows you to follow your runs in real-time, and compare them with each other.

# Frequently Asked Questions

## What is the difference between before and after training metrics?

By default, the 3LC integration collects metrics only after training with the `best.pt` weights written by YOLO. These are the after training metrics.

If a starting metrics collection epoch is provided (optionally with an interval), metrics are also collected during training, this time with the exponential moving average that YOLO uses for its validation passes.

## What happens if I use early stopping? Does it interfere with 3LC?

Early stopping can be used just like before. Unless metrics collection is disabled, final validation passes are performed over the train and validation sets after training, regardless of whether that is due to early stopping or completing all the epochs.

## Why is embeddings collection disabled by default?

Embeddings collection has an extra dependency for the library used for reduction, and a performance implication (fitting and applying the reducer) at the end of a run. It is therefore disabled by default.

## How do I collect embeddings for each bounding box?

In order to collect embeddings (or other additional metrics) for each bounding box, refer to the [advanced 3LC examples](https://github.com/3lc-ai/3lc-examples/tree/main/tutorials/5-advanced-examples).

## Can I use the Ultralytics YOLO CLI commands in the integration to train and collect metrics?

This is not supported yet, but will be added in a future commit!

## Why is the 3LC integration pinned to just a few versions of `Ultralytics`?

Ultralytics makes changes to the internals of the `ultralytics` codebase, which occasionally breaks the 3LC integration. It is therefore pinned to versions which are known to work with the integration.
