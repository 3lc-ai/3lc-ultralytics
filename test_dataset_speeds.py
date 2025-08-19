"""Compare the speed of loading a dataset from 3lc compared to plain ultralytics"""

import cProfile
import pstats
import time
from argparse import Namespace

import tlc
import ultralytics
import ultralytics.data.dataset

import tlc_ultralytics
import tlc_ultralytics.classify.dataset

# Easy toggle switches - just comment/uncomment these lines
PROFILE_YOLO = False  # Set to True to profile YOLO dataset creation
PROFILE_TLC = True  # Set to True to profile TLC dataset creation


def print_diagnostics():
    print(f"ultralytics version: {ultralytics.__version__}")
    print(f"tlc version: {tlc.__version__}")
    # TODO: no __version__ in tlc_ultralytics (only in built wheel?)

    print(f"ultralytics file: {ultralytics.__file__}")
    print(f"tlc_ultralytics file: {tlc_ultralytics.__file__}")
    print(f"tlc file: {tlc.__file__}")


def create_yolo_dataset(yolo_root: str, args: Namespace):
    """Helper function to create YOLO dataset for profiling"""
    return ultralytics.data.dataset.ClassificationDataset(
        root=yolo_root,
        args=args,
    )


def create_tlc_dataset(tlc_table: tlc.Table, args: Namespace):
    """Helper function to create TLC dataset for profiling"""
    return tlc_ultralytics.classify.dataset.TLCClassificationDataset(
        table=tlc_table,
        args=args,
    )


def setup_datasets(
    yolo_root: str,
    tlc_table: tlc.Table,
):
    args = Namespace(cache=False, scale=1.0, pad=False, pad_value=0, imgsz=32)

    yolo_time = None
    tlc_time = None

    # Profile YOLO dataset creation
    if PROFILE_YOLO:
        print("Profiling YOLO dataset creation...")
        yolo_profiler = cProfile.Profile()
        start_time = time.time()
        try:
            yolo_profiler.enable()
            create_yolo_dataset(yolo_root, args)
            yolo_profiler.disable()
            yolo_time = time.time() - start_time
            print(f"YOLO dataset creation time: {yolo_time:.3f} seconds")
        except KeyboardInterrupt:
            yolo_profiler.disable()
            yolo_time = time.time() - start_time
            print(f"\nYOLO dataset creation interrupted after {yolo_time:.3f} seconds")
        finally:
            # Save and print YOLO profiling results
            yolo_profiler.dump_stats("yolo_profile.prof")
            print("Top 15 functions in YOLO dataset creation:")
            yolo_stats = pstats.Stats(yolo_profiler)
            yolo_stats.sort_stats("cumulative").print_stats(15)
            print("-" * 50)

    # Profile TLC dataset creation
    if PROFILE_TLC:
        print("Profiling TLC dataset creation...")
        tlc_profiler = cProfile.Profile()
        start_time = time.time()
        try:
            tlc_profiler.enable()
            create_tlc_dataset(tlc_table, args)
            tlc_profiler.disable()
            tlc_time = time.time() - start_time
            print(f"TLC dataset creation time: {tlc_time:.3f} seconds")
        except KeyboardInterrupt:
            tlc_profiler.disable()
            tlc_time = time.time() - start_time
            print(f"\nTLC dataset creation interrupted after {tlc_time:.3f} seconds")
        finally:
            # Save and print TLC profiling results
            tlc_profiler.dump_stats("tlc_profile.prof")
            print("Top 15 functions in TLC dataset creation:")
            tlc_stats = pstats.Stats(tlc_profiler)
            tlc_stats.sort_stats("cumulative").print_stats(15)
            print("-" * 50)

    # Compare speeds if both were profiled
    if yolo_time is not None and tlc_time is not None:
        print(f"Speed difference: {tlc_time / yolo_time:.2f}x slower (TLC vs YOLO)")


if __name__ == "__main__":
    SPOOR_DATA_ROOT = "C:/Data/spoor/crops_padded_all_verified"
    print_diagnostics()
    try:
        setup_datasets(
            SPOOR_DATA_ROOT + "/train",
            tlc.Table.from_names("initial", "train", "SPOOR-BIRD-CLS"),
        )
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user. Profiling data has been saved.")
        print("Check 'yolo_profile.prof' and/or 'tlc_profile.prof' for detailed analysis.")
