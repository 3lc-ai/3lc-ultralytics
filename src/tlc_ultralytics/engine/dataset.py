from __future__ import annotations

from multiprocessing.pool import ThreadPool
from typing import Any

import tlc
from ultralytics.data.utils import verify_image
from ultralytics.utils import LOGGER, NUM_THREADS, TQDM, colorstr


# Responsible for any generic 3LC dataset handling, such as scanning, caching and adding example ids to each sample
# Assume there is an attribute self.table that is a tlc.Table
class TLCDatasetMixin:
    def _post_init(self):
        self.display_name = self.table.dataset_name

        assert hasattr(self, "table") and isinstance(self.table, tlc.Table), (
            "TLCDatasetMixin requires an attribute `table` which is a tlc.Table."
        )

    def __getitem__(self, index):
        """Get the item at the given index, add the example id to the sample for use in metrics collection."""
        example_id = self._index_to_example_id(index)
        sample = super().__getitem__(index)
        sample["example_id"] = example_id
        return sample

    @staticmethod
    def _absolutize_image_url(image_str: str, table_url: tlc.Url) -> str:
        """Expand aliases in the raw image string and absolutize the URL if it is relative.

        :param image_str: The raw image string to absolutize.
        :param table_url: The table URL to use for absolutization, usually the table whose images are being used.
        :return: The absolutized image string.
        :raises ValueError: If the alias cannot be expanded or the image URL is not a local file path.
        """
        url = tlc.Url(image_str)
        try:
            url = url.expand_aliases(allow_unexpanded=False)
        except ValueError as e:
            msg = f"Failed to expand alias in image_str: {image_str}. "
            msg += "Make sure the alias is spelled correctly and is registered in your configuration."
            raise ValueError(msg) from e

        if url.scheme not in (tlc.Scheme.FILE, tlc.Scheme.RELATIVE):
            msg = f"Image URL {url.to_str()} is not a local file path, it has scheme {url.scheme.value}. "
            msg += "Only local image file paths are supported. If your image URLs are not local, first copy "
            msg += "the images to a local directory and use an alias."
            raise ValueError(msg)

        return url.to_absolute(table_url).to_str()

    def _get_label_from_row(self, im_file: str, row: Any, example_id: int) -> Any:
        raise NotImplementedError("Subclasses must implement this method")

    def _index_to_example_id(self, index: int) -> int:
        raise NotImplementedError("Subclasses must implement this method")

    def _get_rows_from_table(self) -> tuple[list[int], list[str], list[Any]]:
        """Get the rows from the table and return a list of rows, excluding zero weight samples and samples with
        problematic images.

        :return: A list of image paths and labels.
        """
        im_files, labels = [], []
        verified_count, corrupt_count, excluded, msgs = 0, 0, 0, []

        colored_prefix = colorstr(self.prefix + ":")
        desc = f"{colored_prefix} Preparing data from {self.table.url.to_str()}"
        weight_column_name = self.table.weights_column_name

        image_paths = [
            self._absolutize_image_url(row[self._image_column_name], self.table.url) for row in self.table.table_rows
        ]

        image_iterator = (((im_file, None), "") for im_file in image_paths)

        with ThreadPool(NUM_THREADS) as pool:
            verified_count, corrupt_count, excluded_count, msgs = 0, 0, 0, []
            results = pool.imap(func=verify_image, iterable=image_iterator)
            iterator = zip(enumerate(self.table.table_rows), results)
            pbar = TQDM(iterator, desc=desc, total=len(image_paths))

            for (example_id, row), ((im_file, _), verified, corrupt, msg) in pbar:
                # Skip zero-weight rows if enabled
                if self._exclude_zero and row.get(weight_column_name, 1) == 0:
                    excluded_count += 1
                else:
                    if verified:
                        im_files.append(im_file)
                        labels.append(self._get_label_from_row(im_file, row, example_id))
                    if msg:
                        msgs.append(msg)

                    verified_count += verified
                    corrupt_count += corrupt

                exclude_str = f" {excluded_count} excluded" if excluded_count > 0 else ""
                pbar.desc = f"{desc} {verified_count} images, {corrupt_count} corrupt{exclude_str}"

            pbar.close()

        if excluded > 0:
            percentage_excluded = excluded_count / len(self.table) * 100
            LOGGER.info(
                f"{colored_prefix} Excluded {excluded_count} ({percentage_excluded:.2f}% of the table) "
                "zero-weight rows."
            )

        if msgs:
            # Only take first 10 messages if there are more
            truncated = len(msgs) > 10
            msgs_to_show = msgs[:10]

            # Create the message string with truncation notice if needed
            msgs_str = "\n".join(msgs_to_show)
            if truncated:
                msgs_str += f"\n... (showing first 10 of {len(msgs)} messages)"

            percentage_corrupt = corrupt_count / (len(self.table) - excluded) * 100

            verb = "is" if corrupt_count == 1 else "are"
            plural = "s" if corrupt_count != 1 else ""
            LOGGER.warning(
                f"{colored_prefix} There {verb} {corrupt_count} ({percentage_corrupt:.2f}%) corrupt image{plural}:"
                f"\n{msgs_str}"
            )

        return im_files, labels
