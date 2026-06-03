<p align="center">
<img src="https://3lc.ai/wp-content/uploads/2023/09/3LC-Logo_Footer.svg">
</p>

<h1 align="center">3LC YOLO Integration</h1>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/3lc-ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/3lc-ultralytics/)
[![Discord](https://img.shields.io/badge/discord-3LC-5865F2?logo=discord&logoColor=white)](https://discord.gg/fwnwFtfafC)

</div>

<p align="center">
<a href="https://docs.ultralytics.com/">Ultralytics YOLO</a> classification, object detection and segmentation with 3LC integrated.
</p>

## About 3LC

[3LC](https://3lc.ai) is a tool which enables data scientists to improve machine learning models in a data-centric fashion. It collects per-sample predictions and metrics, allows viewing and modifying the dataset in the context of those predictions in the 3LC Dashboard, and rerunning training with the revised dataset.

3LC is free for non-commercial use.

![3LC Dashboard Overview](https://github.com/3lc-ai/3lc-ultralytics/blob/develop/src/tlc_ultralytics/_static/dashboard.png?raw=true)

## Quick Start

### Installation

Install the package and requirements into a virtual environment (Python 3.10–3.13):

```bash
pip install 3lc-ultralytics --extra-index-url https://pypi.3lc.ai/public/repositories/releases-public
```

The `--extra-index-url` is required because `3lc` is published on [3LC's public package index](https://pypi.3lc.ai/public/repositories/releases-public), not on PyPI.

If you use [`uv`](https://docs.astral.sh/uv/), the equivalent command line is:

```bash
uv pip install 3lc-ultralytics \
  --extra-index-url https://pypi.3lc.ai/public/repositories/releases-public \
  --index-strategy unsafe-best-match
```

`--index-strategy unsafe-best-match` is required for `uv` (but not `pip`). See the [GitHub README](https://github.com/3lc-ai/3lc-ultralytics#installation) for details and for the `uv add` / `uv sync` project setup.

### Dataset and Training

The integration is documented on the project [GitHub Page](https://github.com/3lc-ai/3lc-ultralytics), and details how to register datasets and run training.
