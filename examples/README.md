## 3LC YOLO Integration Examples

This directory contains minimal runnable examples per task. Each subfolder includes:

- `create_tables.py`: how to create `tlc.Table`s for the task
- `train.py`: training with 3LC integration
- `collect.py`: metrics collection only

Tasks:

- Classification: `examples/classify/`
- Object Detection: `examples/detect/`
- Segmentation: `examples/segment/`
- Pose Estimation: `examples/pose/`
- Oriented Bounding Boxes: `examples/obb/`

Run an example, e.g. detection training:

```bash
python examples/detect/train.py
```

See the top-level README for full documentation.
