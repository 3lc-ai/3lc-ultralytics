from __future__ import annotations

import tlc
import numpy as np

from ultralytics.data.utils import verify_image
from ultralytics.utils import LOGGER, TQDM, colorstr

from typing import Any


# Responsible for any generic 3LC dataset handling, such as scanning, caching and adding example ids to each sample
# Assume there is an attribute self.table that is a tlc.Table
class TLCDatasetMixin:
    def _post_init(self):
        self.display_name = self.table.dataset_name

        assert hasattr(self, "table") and isinstance(self.table, tlc.Table), (
            "TLCDatasetMixin requires an attribute `table` which is a tlc.Table."
        )
        if not hasattr(self, "example_ids"):
            self.example_ids = np.arange(len(self.table))

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample[tlc.EXAMPLE_ID] = self.example_ids[index]  # Add example id to the sample dict
        return sample

    def __len__(self):
        return len(self.example_ids)

    @staticmethod
    def _absolutize_image_url(image_str: str, table_url: tlc.Url) -> str:
        """Expand aliases in the raw image string and absolutize the URL if it is relative.

        :param image_str: The raw image string to absolutize.
        :param table_url: The table URL to use for absolutization, usually the table whose images are being used.
        :return: The absolutized image string.
        :raises ValueError: If the alias cannot be expanded.
        """
        url = tlc.Url(image_str)
        try:
            url = url.expand_aliases(allow_unexpanded=False)
        except ValueError as e:
            raise ValueError(
                f"Failed to expand alias in image_str: {image_str}. "
                "Make sure the alias is spelled correctly and is registered in your configuration."
            ) from e

        return url.to_absolute(table_url).to_str()

    def _get_label_from_row(self, im_file: str, row: Any, example_id: int) -> Any:
        raise NotImplementedError("Subclasses must implement this method")

    def _get_rows_from_table(self) -> tuple[list[int], list[str], list[Any]]:
        """Get the rows from the table and return a list of rows, excluding zero weight samples and samples with
        problematic images.

        :return: A list of example ids, image paths and labels.
        """

        example_ids, im_files, labels = [], [], []

        nf, nc, excluded, msgs = 0, 0, 0, []
        colored_prefix = colorstr(self.prefix + ":")
        desc = f"{colored_prefix} Preparing data from {self.table.url.to_str()}"
        pbar = TQDM(enumerate(self.table.table_rows), desc=desc, total=len(self.table))

        weight_column_name = self.table.weights_column_name

        for example_id, row in pbar:
            if self._exclude_zero and row.get(weight_column_name, 1) == 0:
                excluded += 1
                continue

            im_file = self._absolutize_image_url(row[self._image_column_name], self.table.url)

            (im_file, _), nf_f, nc_f, msg = verify_image(((im_file, None), ""))

            nf += nf_f
            nc += nc_f

            if nc_f:
                msgs.append(msg)
                continue

            example_ids.append(example_id)
            im_files.append(im_file)
            labels.append(self._get_label_from_row(im_file, row, example_id))

            exclude_str = f" {excluded} excluded" if excluded > 0 else ""
            pbar.desc = f"{desc} {nf} images, {nc} corrupt{exclude_str}"

        pbar.close()

        if excluded > 0:
            percentage_excluded = excluded / len(self.table) * 100
            LOGGER.info(
                f"{colored_prefix} Excluded {excluded} ({percentage_excluded:.2f}% of the table) zero-weight rows."
            )

        if msgs:
            # Only take first 10 messages if there are more
            truncated = len(msgs) > 10
            msgs_to_show = msgs[:10]

            # Create the message string with truncation notice if needed
            msgs_str = "\n".join(msgs_to_show)
            if truncated:
                msgs_str += f"\n... (showing first 10 of {len(msgs)} messages)"

            percentage_corrupt = nc / (len(self.table) - excluded) * 100

            verb = "is" if nc == 1 else "are"
            LOGGER.warning(
                f"{colored_prefix} There {verb} {nc} ({percentage_corrupt:.2f}%) corrupt image{'' if nc == 1 else 's'}:"
                f"\n{msgs_str}"
            )

        return example_ids, im_files, labels
