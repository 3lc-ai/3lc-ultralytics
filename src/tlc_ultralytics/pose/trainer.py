from __future__ import annotations

from copy import deepcopy

from ultralytics.models.yolo.pose.train import PoseTrainer

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME, POSE_LABEL_COLUMN_NAME
from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.pose.utils import tlc_check_pose_dataset
from tlc_ultralytics.pose.validator import TLCPoseValidator


class TLCPoseTrainer(PoseTrainer, TLCDetectionTrainer):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = POSE_LABEL_COLUMN_NAME

    def get_dataset(self):
        tables = self._tables
        random_table = next(iter(tables.values()))
        self.data = {
            **tables,
            "names": {0: "person"},
            "names_3lc": {"person": 0},
            "nc": 1,
            "range_to_3lc_class": {0: 0},
            "3lc_class_to_range": {0: 0},
            "channels": 3,  # TODO(Frederik): Read out channels from appropriate place and populate here
            "kpt_shape": random_table.kpt_shape if hasattr(random_table, "kpt_shape") else (17, 3),
            "flip_idx": random_table.flip_idx if hasattr(random_table, "flip_idx") else None,
        }

        if "val" not in self.data:
            data_test = tlc_check_pose_dataset(
                self.args.data,
                self._tables,
                self._image_column_name,
                self._label_column_name,
                project_name=self._settings.project_name,
                splits=("test",),
            )
            self.data["test"] = data_test["test"]

        return self.data

    def get_validator(self, dataloader=None):
        self.loss_names = ("box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss")
        if not dataloader:
            dataloader = self.test_loader

        return TLCPoseValidator(
            dataloader,
            save_dir=self.save_dir,
            args=deepcopy(self.args),
            _callbacks=self.callbacks,
            run=self._run,
            settings=self._settings,
            training=True,
        )

    # def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
    #     """Construct and return dataloader."""

    #     sampler = create_sampler(dataset_path, mode, self._settings, distributed=rank != -1)

    #     # Patch parent class module to use our build_dataloader
    #     trainer_build_dataloader = ultralytics.models.yolo.detect.train.build_dataloader
    #     ultralytics.models.yolo.detect.train.build_dataloader = partial(build_dataloader, sampler=sampler)

    #     dataloader = super().get_dataloader(dataset_path, batch_size, rank, mode)

    #     # Restore parent class module
    #     ultralytics.models.yolo.detect.train.build_dataloader = trainer_build_dataloader

    #     return dataloader

    # def build_dataset(self, *args, **kwargs):
    #     from ultralytics.models.yolo.detect.train import build_yolo_dataset as original_build_yolo_dataset

    #     mode = kwargs.get("mode") or args[1]

    #     exclude_zero = mode == "val" and self._settings.exclude_zero_weight_collection
    #     ultralytics.models.yolo.detect.train.build_yolo_dataset = partial(
    #         build_tlc_yolo_dataset,
    #         exclude_zero=exclude_zero,
    #         class_map=self.data["3lc_class_to_range"],
    #         image_column_name=self._image_column_name,
    #         label_column_name=self._label_column_name,
    #     )

    #     result = PoseTrainer.build_dataset(self, *args, **kwargs)

    #     ultralytics.models.yolo.detect.train.build_yolo_dataset = original_build_yolo_dataset

    #     return result
