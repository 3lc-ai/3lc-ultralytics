# 3LC YOLO Integration Examples

This directory contains examples for using the 3LC YOLO integration for different computer vision tasks. For the main documentation, see the [repository README](../README.md).

## Example Files

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

After running the examples, you can:

1. **View results in the 3LC Dashboard**: Open the generated runs to explore metrics and visualizations
1. **Modify your data**: Use the dashboard to identify and fix issues in your dataset
1. **Retrain with improved data**: Use the updated tables for better model performance

For more advanced usage and configuration options, refer to the [main README](../README.md) and the [3LC Settings section](../README.md#3lc-settings).
