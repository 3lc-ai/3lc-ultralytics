# 3LC YOLO Integration Examples

This directory contains examples for using the 3LC YOLO integration for different computer vision tasks. For the main documentation, see the [repository README](../README.md).

## Register your dataset - Creating `tlc.Table`s

The first step when working with 3LC is to register your dataset as a 3LC Table. Typically, a `tlc.Table` is created for each split of your dataset. The way to do this is different for each task, and we will show these here. There are several ways of creating `tlc.Table`s, such as using a `tlc.TableWriter`, but most of the time the easiest is to use convenience methods that are available for common dataset formats. We will show the ones that are natively used in Ultralytics YOLO here.

### Classification

The dataset format used in YOLO for classification is the `ImageFolder` available in `torchvision`. Check out ... for more details. To create a `tlc.Table` for an ImageFolder dataset, the following code can be used:

TODO(Frederik): improve

```python
import tlc

table = tlc.Table.from_image_folder(...)
```

### Object Detection

### Instance Segmentation

## Training

To run training with the 3LC integration, the 

## Collection

The examples are grouped by task, 

### Classification

- **[train.py](classify/train.py)**: Basic classification training example using the MNIST dataset.
- **[collect.py](classify/collect.py)**: Metrics collection on the Imagenet dataset.

### Object Detection

- **[train.py](detect/train.py)**: Basic object detection training example using a small Signatures dataset.
- **[collect.py](detect/collect.py)**: Metrics collection example for object detection on COCO128.

### Segmentation

- **[train.py](segment/train.py)**: Basic instance segmentation example on the Carparts dataset.
- **[collect.py](segment/collect.py)**: Metrics collection example for object detection on COCO128-seg.

## Using the Examples

1. **Install the integration**: Follow the installation instructions in the [main README](../README.md)
1. **Run the examples**: Execute the example files to see how to use the integration:

```bash
# Classification
python examples/classify/train.py

# Object Detection
python examples/detect/train.py
python examples/detect/collect.py
```

## Creating Tables for Different Tasks

The 3LC YOLO integration supports three main tasks:

- **Classification**: Image classification with `tlc.Table.from_image_folder(...)`.
- **Object Detection**: Bounding box detection with `tlc.Table.from_yolo(..., task="detect")`.
- **Segmentation**: Instance segmentation with `tlc.Table.from_yolo(..., task="segment")`.

## Next Steps

After running the examples, you can iteratively:

1. **View results in the 3LC Dashboard**: Open the generated runs to explore metrics and visualizations.
1. **Modify your data**: Use the Dashboard to debug issues in your dataset, creating new edited `Table`s.
1. **Retrain with improved data**: Use the edited `Table`s in a new training run for better model performance.

For more advanced usage and configuration options, refer to the [main README](../README.md) and the [3LC Settings section](../README.md#3lc-settings).
