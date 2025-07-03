from __future__ import annotations

import hashlib
import json
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

        if url.scheme is not tlc.Scheme.FILE:
            msg = f"Image URL {url.to_str()} is not a local file path, it has scheme {url.scheme.value}. "
            msg += "Only local image file paths are supported. If your image URLs are not local, first copy "
            msg += "the images to a local directory and use an alias."
            raise ValueError(msg)

        return url.to_absolute(table_url).to_str()

    def _get_label_from_row(self, im_file: str, row: Any, example_id: int) -> Any:
        raise NotImplementedError("Subclasses must implement this method")

    def _index_to_example_id(self, index: int) -> int:
        raise NotImplementedError("Subclasses must implement this method")

    def _get_cache_key(self, image_paths: list[str], exclude_zero: bool) -> str:
        """Generate a cache key based on the hash of all image paths and whether to exclude zero weight samples.

        :param image_paths: List of absolute image paths
        :param exclude_zero: Whether to exclude zero weight samples
        :return: Cache key string
        """
        # Sort paths to ensure consistent hash regardless of order
        sorted_paths = sorted(image_paths)

        # Create hash of all paths concatenated
        paths_str = "".join(sorted_paths)

        # Add exclude_zero flag to the hash
        paths_str += f"exclude_zero={exclude_zero}"
        return hashlib.md5(paths_str.encode()).hexdigest()

    def _get_cache_path(self, table_url: tlc.Url, cache_key: str) -> tlc.Url:
        """Get the URL to the cache file, in the same directory as the table.

        :param table_url: The table URL to use for the cache path
        :param cache_key: The cache key to use for the cache path
        :return: The URL to the cache file
        """
        return table_url / f"yolo_{cache_key}.json"

    def _encode_example_ids(self, example_ids: list[int]) -> list[dict[str, int]]:
        """Encode a list of example IDs into ranges for efficient storage.

        :param example_ids: List of example IDs (assumed to be sorted)
        :return: List of range dictionaries with 'start' and 'end' keys
        """
        if not example_ids:
            return []

        ranges = []
        start = example_ids[0]
        end = start

        for i in range(1, len(example_ids)):
            if example_ids[i] == end + 1:
                # Consecutive, extend current range
                end = example_ids[i]
            else:
                # Gap found, save current range and start new one
                ranges.append({"start": start, "end": end})
                start = example_ids[i]
                end = start

        # Don't forget the last range
        ranges.append({"start": start, "end": end})

        return ranges

    def _decode_example_ids(self, ranges: list[dict[str, int]]) -> list[int]:
        """Decode ranges back into a list of example IDs.

        :param ranges: List of range dictionaries with 'start' and 'end' keys
        :return: List of example IDs
        """
        example_ids = []
        for range_dict in ranges:
            start = range_dict["start"]
            end = range_dict["end"]
            example_ids.extend(range(start, end + 1))
        return example_ids

    def _load_cached_example_ids(self, cache_url: tlc.Url) -> list[int] | None:
        """Load the cached example ids from the cache file.

        :param cache_path: The path to the cache file
        :return: A list of example ids
        """
        try:
            cache_data = json.loads(cache_url.read(mode="s"))

            # Check cache version
            if cache_data.get("version") != 1:
                LOGGER.info("Cache version mismatch, regenerating cache.")
                return None

            if "ranges" in cache_data:
                return self._decode_example_ids(cache_data["ranges"])
            else:
                LOGGER.warning("Cache file missing ranges field, regenerating cache.")
                return None

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            LOGGER.warning(f"Failed to load cache: {e}, regenerating cache.")
            return None

    def _save_cached_example_ids(self, cache_url: tlc.Url, example_ids: list[int]):
        """Save the example ids to the cache file.

        :param cache_url: The URL to the cache file
        :param example_ids: A list of example ids
        """
        ranges = self._encode_example_ids(example_ids)

        content = {
            "version": 1,
            "ranges": ranges,
        }
        cache_url.write(json.dumps(content, indent=2), mode="s")

    def _get_rows_from_table(self) -> tuple[list[str], list[Any]]:
        """Get the rows from the table and return a list of example ids, excluding zero weight and corrupt images.
        Rely on the cache to avoid recomputing example ids if possible.

        :return: A list of image paths and labels.
        """

        image_paths = [
            self._absolutize_image_url(row[self._image_column_name], self.table.url) for row in self.table.table_rows
        ]

        cache_key = self._get_cache_key(image_paths, self._exclude_zero)
        cache_path = self._get_cache_path(self.table.url, cache_key)

        example_ids = self._load_cached_example_ids(cache_path) if cache_path.exists() else None

        if example_ids is not None:
            LOGGER.info(f"{colorstr(self.prefix)}: Loaded {len(example_ids)} samples from cache.")

        if example_ids is None:
            example_ids = self._get_example_ids_from_table(image_paths)
            self._save_cached_example_ids(cache_path, example_ids)

        im_files = [image_paths[i] for i in example_ids]
        labels = [self._get_label_from_row(im_file, self.table.table_rows[i], i) for i, im_file in enumerate(im_files)]

        return im_files, labels

    def _get_example_ids_from_table(self, image_paths: list[str]) -> list[int]:
        """Get the example ids to use from the table, excluding zero weight samples and samples with problematic images.

        :param image_paths: List of absolute image paths
        :return: A list of example ids
        """
        example_ids = []
        verified_count, corrupt_count, excluded_count, msgs = 0, 0, 0, []
        colored_prefix = colorstr(self.prefix + ":")
        desc = f"{colored_prefix} Preparing data from {self.table.url.to_str()}"
        weight_column_name = self.table.weights_column_name

        image_iterator = (((im_file, None), "") for im_file in image_paths)

        with ThreadPool(NUM_THREADS) as pool:
            verified_count, corrupt_count, excluded_count, msgs = 0, 0, 0, []
            results = pool.imap(func=verify_image, iterable=image_iterator)
            iterator = zip(enumerate(self.table.table_rows), results)
            pbar = TQDM(iterator, desc=desc, total=len(image_paths))

            for (example_id, row), (_, verified, corrupt, msg) in pbar:
                # Skip zero-weight rows if enabled
                if self._exclude_zero and row.get(weight_column_name, 1) == 0:
                    excluded_count += 1
                else:
                    if verified:
                        example_ids.append(example_id)
                    if msg:
                        msgs.append(msg)

                    verified_count += verified
                    corrupt_count += corrupt

                exclude_str = f" {excluded_count} excluded" if excluded_count > 0 else ""
                pbar.desc = f"{desc} {verified_count} images, {corrupt_count} corrupt{exclude_str}"

            pbar.close()

        if excluded_count > 0:
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

            percentage_corrupt = corrupt_count / (len(self.table) - excluded_count) * 100

            verb = "is" if corrupt_count == 1 else "are"
            plural = "s" if corrupt_count != 1 else ""
            LOGGER.warning(
                f"{colored_prefix} There {verb} {corrupt_count} ({percentage_corrupt:.2f}%) corrupt image{plural}:"
                f"\n{msgs_str}"
            )

        return example_ids
