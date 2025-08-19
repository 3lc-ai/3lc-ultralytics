# python examples/classify/train_spoor.py 2>&1 | tee logs/training_output_debug.log

# logger.info(f"TableWriter: total allocated memory: {pa.total_allocated_bytes() / 1024 / 1024:.1f} MB")

import gc
import logging
import threading
import time
from functools import wraps

import psutil
import pyarrow as pa
import tlc
import torch

from tlc_ultralytics import YOLO, Settings


# MONKEY PATCH TO SKIP TRAINING WORK FOR FASTER VALIDATION DEBUGGING
def patch_trainer_for_validation_debugging():
    """Patch the BaseTrainer to skip actual training work but keep validation flow"""
    from ultralytics.engine.trainer import BaseTrainer

    # Store original training method
    original_do_train = BaseTrainer._do_train

    # Apply patches
    def apply_patches():
        def patched_do_train(self, world_size=1):
            """Modified training loop that skips actual computation but keeps validation"""
            import math
            import time
            import warnings

            from ultralytics.utils import LOGGER, RANK, TQDM, colorstr

            if world_size > 1:
                self._setup_ddp(world_size)
            self._setup_train(world_size)

            nb = len(self.train_loader)  # number of batches
            self.epoch_time = None
            self.epoch_time_start = time.time()
            self.train_time_start = time.time()
            self.run_callbacks("on_train_start")

            LOGGER.info(
                f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
                f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
                f"Logging results to {colorstr('bold', self.save_dir)}\n"
                f"Starting FAST DEBUGGING training for "
                + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
            )

            epoch = self.start_epoch
            self.optimizer.zero_grad()

            while True:
                self.epoch = epoch
                self.run_callbacks("on_train_epoch_start")

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.scheduler.step()

                self._model_train()
                if RANK != -1:
                    self.train_loader.sampler.set_epoch(epoch)

                if RANK in {-1, 0}:
                    LOGGER.info(self.progress_string())

                # FAST BATCH LOOP - Skip actual training computation
                # Create dummy loss values for the epoch
                dummy_loss = torch.tensor(0.1, device=self.device, requires_grad=True)
                dummy_loss_items = torch.tensor([0.1], device=self.device)
                self.tloss = dummy_loss_items
                self.loss = dummy_loss
                self.loss_items = dummy_loss_items

                # Fake progress through batches quickly
                batch_count = 0
                for i, batch in enumerate(self.train_loader):
                    self.run_callbacks("on_train_batch_start")

                    # Skip all the actual work (forward/backward/optimizer)
                    # Just pretend we processed the batch

                    if RANK in {-1, 0}:
                        print(f"Skipping batch {i + 1}/{nb} (epoch {epoch + 1})", end="\r")

                    self.run_callbacks("on_train_batch_end")
                    batch_count += 1

                    # Break early to speed up even more (only process a few batches)
                    if batch_count >= 3:
                        break

                # Continue with validation and epoch end logic (IMPORTANT PART)
                self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
                self.run_callbacks("on_train_epoch_end")

                if RANK in {-1, 0}:
                    final_epoch = epoch + 1 >= self.epochs
                    self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                    # VALIDATION - This runs normally (where your bug is!)
                    if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                        print(f"\n🔍 RUNNING VALIDATION (epoch {epoch + 1}) - Bug should occur here!")
                        self.metrics, self.fitness = self.validate()

                    self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                    self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch

                    if self.args.time:
                        self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                    # Save model
                    if self.args.save or final_epoch:
                        self.save_model()
                        self.run_callbacks("on_model_save")

                # Scheduler and timing
                t = time.time()
                self.epoch_time = t - self.epoch_time_start
                self.epoch_time_start = t

                self.run_callbacks("on_fit_epoch_end")

                if self._get_memory(fraction=True) > 0.5:
                    self._clear_memory()

                # Early stopping logic
                if RANK != -1:
                    from torch import distributed as dist

                    broadcast_list = [self.stop if RANK == 0 else None]
                    dist.broadcast_object_list(broadcast_list, 0)
                    self.stop = broadcast_list[0]

                if self.stop:
                    break
                epoch += 1

            # Final evaluation
            if RANK in {-1, 0}:
                seconds = time.time() - self.train_time_start
                LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
                self.final_eval()
                if self.args.plots:
                    self.plot_metrics()
                self.run_callbacks("on_train_end")

            self._clear_memory()
            from ultralytics.utils.torch_utils import unset_deterministic

            unset_deterministic()
            self.run_callbacks("teardown")

        # Replace the training method
        BaseTrainer._do_train = patched_do_train

        print("🔧 TRAINING PATCHES APPLIED - Training work disabled, validation will run normally!")
        print("   ⚡ Training batches: FAST SKIP (only 3 batches per epoch)")
        print("   ⚡ Forward/Backward: DISABLED")
        print("   ⚡ Optimizer steps: DISABLED")
        print("   ✅ Validation: ENABLED (your bug should trigger here)")
        print("   ✅ Callbacks & Metrics: ENABLED")

    return apply_patches


# Apply the patches before any model/trainer creation
# Comment out these lines to disable the patch and run normal training
patch_training = patch_trainer_for_validation_debugging()
patch_training()

# logging.getLogger("tlc").setLevel(logging.DEBUG)
# logging.getLogger("tlc").handlers = []
# logging.getLogger("tlc").addHandler(logging.FileHandler("logs/debug-spoor-3lc.log", mode="w"))

SPOOR_DATA_ROOT = "C:/Data/spoor/crops_padded_all_verified"
PROJECT_NAME = "SPOOR-BIRD-CLS"

tlc.register_project_url_alias("SPOOR_BIRD_CLS_DATA", SPOOR_DATA_ROOT, project=PROJECT_NAME)


def get_memory_info():
    """Get comprehensive memory information"""
    process = psutil.Process()
    memory_info = process.memory_info()

    info = {
        "cpu_memory_mb": memory_info.rss / 1024 / 1024,
        "cpu_memory_percent": process.memory_percent(),
        "available_memory_mb": psutil.virtual_memory().available / 1024 / 1024,
        "total_memory_mb": psutil.virtual_memory().total / 1024 / 1024,
        "pyarrow_memory_mb": pa.total_allocated_bytes() / 1024 / 1024,
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
    print(f"PyArrow Memory: {info['pyarrow_memory_mb']:.1f} MB")

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

    def __init__(self, interval=60):
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
            print(
                f"\n[MONITOR] CPU: {info['cpu_memory_mb']:.1f}MB ({info['cpu_memory_percent']:.1f}%) | PyArrow: {info['pyarrow_memory_mb']:.1f}MB",
                end="",
            )
            if torch.cuda.is_available():
                print(
                    f" | GPU: {info['gpu_memory_allocated_mb']:.1f}MB ({(info['gpu_memory_allocated_mb'] / info['gpu_memory_total_mb'] * 100):.1f}%)"
                )
            else:
                print()
            print(f"Len of object registry: {len(tlc.ObjectRegistry._objects)}")
            time.sleep(self.interval)


def create_tables():
    tables = {}

    for split, folder in [("train", "train"), ("val", "validation"), ("test", "test")]:
        table = tlc.Table.from_image_folder(
            root=SPOOR_DATA_ROOT + "/" + folder,
            table_name="initial",
            dataset_name=split,
            project_name=PROJECT_NAME,
            if_exists="reuse",
        ).revision(table_name="subset_10")  # .latest()  # Use "revision" or "latest" to get the table you want

        # Create a subset of the table (only needs to run once)
        # table = tlc.SubsetTable(
        #     url=table.url.create_sibling("subset_75"),
        #     input_table_url=table.url,
        #     range_factor_min=0,
        #     range_factor_max=1,
        #     include_probability=0.75,
        # )
        # print(f"Created subset with {len(table)} samples")
        tables[split] = table  # or table!

    return tables


def create_tables_from_names():
    tables = {
        "train": tlc.Table.from_names("initial", "train", PROJECT_NAME),
        "val": tlc.Table.from_names("initial", "val", PROJECT_NAME),
        "test": tlc.Table.from_names("initial", "test", PROJECT_NAME),
    }

    return tables


if __name__ == "__main__":
    # Start background monitoring
    monitor = MemoryMonitor(interval=600)  # Log every 1 minutes
    monitor.start()

    try:
        # Model loading
        model = YOLO("yolo11n-cls.pt")
        # model = YOLO("runs/classify/train13/weights/best.pt")

        # Settings creation
        settings = Settings(
            project_name=PROJECT_NAME,
            image_embeddings_dim=3,
            run_description="MEMORY DEBUGGING",
            collection_epoch_start=1,
            collection_epoch_interval=1,
        )

        # Table creation
        tables = create_tables_from_names()

        # Training start
        model.train(
            tables=tables,
            settings=settings,
            epochs=10,
            imgsz=32,
            workers=4,
            batch=32,
        )
        # log_memory("After training complete")

    except Exception as e:
        log_memory("EXCEPTION OCCURRED", force_gc=True)
        print(f"Exception: {e}")
        raise
    finally:
        monitor.stop()
        log_memory("SCRIPT END", force_gc=True)
        assert True


"""
10%:
50%:
75%:
"""
