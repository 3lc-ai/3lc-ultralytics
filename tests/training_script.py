#!/usr/bin/env python3
"""
Training script to be run in subprocesses to avoid global random state pollution.
This script runs either ultralytics or 3LC training based on command line arguments.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import tlc_ultralytics
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tlc
from ultralytics.models.yolo import YOLO
from tlc_ultralytics import Settings, YOLO as TLCYOLO


def setup_tlc_config(tmp_dir: Path):
    """Setup 3LC configuration for the subprocess"""
    tmp_project_root_url = tlc.Url(tmp_dir / "3LC")
    tlc.Configuration.instance().project_root_url = tmp_project_root_url
    tlc.TableIndexingTable.instance().add_scan_url(
        {
            "url": tlc.Url(tmp_project_root_url),
            "layout": "project",
            "object_type": "table",
            "static": True,
        }
    )


def reset_random_state(seed: int):
    """Reset all random generators to a specific seed."""
    import random
    import numpy as np
    import torch

    # Reset Python's random module
    random.seed(seed)

    # Reset numpy's random state
    np.random.seed(seed)
    np.random.default_rng(seed)

    # Reset PyTorch's random state
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set PyTorch to deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variables for determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Reset any global random state that might persist
    if hasattr(torch, "generator"):
        torch.generator.manual_seed(seed)

    # Additional PyTorch random state resets
    if hasattr(torch, "get_rng_state"):
        torch.set_rng_state(torch.manual_seed(seed).get_state())

    # Force garbage collection to clear any cached random state
    import gc

    gc.collect()


def run_ultralytics_training(task: str, overrides: dict, output_dir: Path):
    """Run ultralytics training and save results"""
    task2model = {"detect": "yolo11n.pt", "classify": "yolo11n-cls.pt", "segment": "yolo11n-seg.pt"}

    model = YOLO(task2model[task])
    results = model.train(**overrides)

    # Save results to output directory
    results_file = output_dir / "ultralytics_results.json"
    with open(results_file, "w") as f:
        json.dump(results.results_dict, f, indent=2)

    # Save names to output directory
    names_file = output_dir / "ultralytics_names.json"
    with open(names_file, "w") as f:
        json.dump(results.names, f, indent=2)

    print(f"Ultralytics training completed. Results saved to {output_dir}")


def run_3lc_training(task: str, overrides: dict, settings: Settings, output_dir: Path):
    """Run 3LC training and save results"""
    task2model = {"detect": "yolo11n.pt", "classify": "yolo11n-cls.pt", "segment": "yolo11n-seg.pt"}

    model = TLCYOLO(task2model[task])
    results = model.train(**overrides, settings=settings)

    # Save results to output directory
    results_file = output_dir / "3lc_results.json"
    with open(results_file, "w") as f:
        json.dump(results.results_dict, f, indent=2)

    # Save names to output directory
    names_file = output_dir / "3lc_names.json"
    with open(names_file, "w") as f:
        json.dump(results.names, f, indent=2)

    print(f"3LC training completed. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run training in isolated subprocess")
    parser.add_argument("--task", required=True, choices=["detect", "classify", "segment"])
    parser.add_argument("--mode", required=True, choices=["ultralytics", "3lc"])
    parser.add_argument("--overrides", required=True, help="JSON string of training overrides")
    parser.add_argument("--settings", help="JSON string of 3LC settings (only for 3lc mode)")
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    parser.add_argument("--tmp-dir", required=True, help="Temporary directory for 3LC")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup 3LC configuration
    setup_tlc_config(Path(args.tmp_dir))

    # Reset random state
    reset_random_state(args.seed)

    # Parse overrides
    overrides = json.loads(args.overrides)

    if args.mode == "ultralytics":
        run_ultralytics_training(args.task, overrides, output_dir)
    elif args.mode == "3lc":
        if not args.settings:
            raise ValueError("Settings required for 3LC mode")
        settings_dict = json.loads(args.settings)
        # Filter out internal attributes that shouldn't be passed to Settings constructor
        filtered_settings = {k: v for k, v in settings_dict.items() if not k.startswith("_")}
        settings = Settings(**filtered_settings)
        run_3lc_training(args.task, overrides, settings, output_dir)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
