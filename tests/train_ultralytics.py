from ultralytics import YOLO
from random_tracker import start_tracking, stop_tracking, print_random_call_count, get_call_details
import json
import os

if __name__ == "__main__":
    # Start tracking random calls before any imports or operations
    start_tracking()

    model = YOLO("yolo11n.pt")

    model.train(
        data="coco8.yaml",
        epochs=1,
        batch=4,
        imgsz=64,
        device="cpu",
        workers=0,
        seed=42,
        deterministic=True,
        save=False,
        plots=False,
        project="tests/tmp/ultralytics",
        name="train_detect",
    )

    # Stop tracking and print results
    stop_tracking()
    print_random_call_count("train_ultralytics.py")

    # Save call details for comparison
    call_details = get_call_details()
    os.makedirs("tmp", exist_ok=True)
    with open("tmp/ultralytics_random_calls.json", "w") as f:
        json.dump(call_details, f, indent=2)
