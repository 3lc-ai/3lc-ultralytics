# python examples/classify/train_spoor_no_3lc.py 2>&1 | tee logs/training_output_no_3lc.log
import gc
import threading
import time
from functools import wraps

import psutil
import torch
from ultralytics import YOLO

SPOOR_DATA_ROOT = "C:/Data/spoor/crops_padded_all_verified"


def get_memory_info():
    """Get comprehensive memory information"""
    process = psutil.Process()
    memory_info = process.memory_info()

    info = {
        "cpu_memory_mb": memory_info.rss / 1024 / 1024,
        "cpu_memory_percent": process.memory_percent(),
        "available_memory_mb": psutil.virtual_memory().available / 1024 / 1024,
        "total_memory_mb": psutil.virtual_memory().total / 1024 / 1024,
    }

    # Add GPU memory info if available
    if torch.cuda.is_available():
        info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        info["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

    return info


def log_memory(stage="", force_gc=False):
    """Log memory usage with optional garbage collection"""
    if force_gc:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    info = get_memory_info()
    print(f"\n{'=' * 60}")
    print(f"MEMORY DEBUG - {stage}")
    print(f"{'=' * 60}")
    print(f"CPU Memory: {info['cpu_memory_mb']:.1f} MB ({info['cpu_memory_percent']:.1f}%)")
    print(f"Available Memory: {info['available_memory_mb']:.1f} MB")
    print(f"Total Memory: {info['total_memory_mb']:.1f} MB")

    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {info['gpu_memory_allocated_mb']:.1f} MB")
        print(f"GPU Memory Reserved: {info['gpu_memory_reserved_mb']:.1f} MB")
        print(f"GPU Memory Total: {info['gpu_memory_total_mb']:.1f} MB")
        print(f"GPU Memory Usage: {(info['gpu_memory_allocated_mb'] / info['gpu_memory_total_mb'] * 100):.1f}%")

    print(f"{'=' * 60}\n")
    return info


def memory_monitor_decorator(func):
    """Decorator to monitor memory usage around function calls"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        log_memory(f"BEFORE {func.__name__}")
        try:
            result = func(*args, **kwargs)
            log_memory(f"AFTER {func.__name__}")
            return result
        except Exception as e:
            log_memory(f"ERROR in {func.__name__}", force_gc=True)
            raise e

    return wrapper


class MemoryMonitor:
    """Background memory monitoring thread"""

    def __init__(self, interval=600):
        self.interval = interval
        self.running = False
        self.thread = None

    def start(self):
        """Start background monitoring"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
        print(f"Started background memory monitoring (interval: {self.interval}s)")

    def stop(self):
        """Stop background monitoring"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("Stopped background memory monitoring")

    def _monitor(self):
        """Background monitoring loop"""
        while self.running:
            info = get_memory_info()
            print(f"\n[MONITOR] CPU: {info['cpu_memory_mb']:.1f}MB ({info['cpu_memory_percent']:.1f}%)", end="")
            if torch.cuda.is_available():
                print(
                    f" | GPU: {info['gpu_memory_allocated_mb']:.1f}MB ({(info['gpu_memory_allocated_mb'] / info['gpu_memory_total_mb'] * 100):.1f}%)"
                )
            else:
                print()
            time.sleep(self.interval)


if __name__ == "__main__":
    # Initial memory state
    log_memory("SCRIPT START", force_gc=True)

    # Start background monitoring
    monitor = MemoryMonitor(interval=30)  # Log every 30 seconds
    monitor.start()

    try:
        # Model loading
        log_memory("Before model loading")
        model = YOLO("yolo11n-cls.pt")
        log_memory("After model loading")

        # Training start
        log_memory("Before training start")
        model.train(
            data=SPOOR_DATA_ROOT,
            epochs=5,
            imgsz=32,
            workers=8,
        )
        log_memory("After training complete")

    except Exception as e:
        log_memory("EXCEPTION OCCURRED", force_gc=True)
        print(f"Exception: {e}")
        raise
    finally:
        monitor.stop()
        log_memory("SCRIPT END", force_gc=True)
