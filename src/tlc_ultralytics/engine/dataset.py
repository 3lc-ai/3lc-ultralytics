from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
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

    def _get_cache_key(self, image_paths: list[str]) -> str:
        """Generate a cache key based on the hash of all image paths.

        :param image_paths: List of absolute image paths
        :return: Cache key string
        """
        # Sort paths to ensure consistent hash regardless of order
        sorted_paths = sorted(image_paths)

        # Create hash of all paths concatenated
        paths_str = "".join(sorted_paths)
        return hashlib.md5(paths_str.encode()).hexdigest()

    def _get_cache_path(self, table_url: tlc.Url, cache_key: str) -> tlc.Url:
        """Get the URL to the cache file, in the same directory as the table.

        :param table_url: The table URL to use for the cache path
        :param cache_key: The cache key to use for the cache path
        :return: The URL to the cache file
        """
        return table_url / f"yolo_{cache_key}.json"

    def _load_cached_example_ids(self, cache_url: tlc.Url) -> list[int] | None:
        """Load the cached corrupt example ids from the cache file.

        :param cache_url: The path to the cache file
        :return: A list of corrupt example ids, or None if cache is invalid
        """
        try:
            cache_data = json.loads(cache_url.read(mode="s"))

            # Check cache version
            if cache_data.get("version") != 1:
                LOGGER.info("Cache version mismatch, regenerating cache.")
                return None

            if "corrupt_example_ids" not in cache_data:
                LOGGER.warning("Cache file missing corrupt_example_ids field, regenerating cache.")
                return None

            # Get corrupt example IDs
            corrupt_example_ids = cache_data["corrupt_example_ids"]
            return corrupt_example_ids

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            LOGGER.warning(f"Failed to load cache: {e}, regenerating cache.")
            return None

    def _save_cached_example_ids(self, cache_url: tlc.Url, corrupt_example_ids: list[int]):
        """Save the corrupt example ids to the cache file.

        :param cache_url: The URL to the cache file
        :param corrupt_example_ids: A list of corrupt example ids
        """
        content = {
            "version": 1,
            "corrupt_example_ids": corrupt_example_ids,
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

        cache_key = self._get_cache_key(image_paths)
        cache_path = self._get_cache_path(self.table.url, cache_key)

        corrupt_example_ids = self._load_cached_example_ids(cache_path) if cache_path.exists() else None

        if corrupt_example_ids is not None:
            LOGGER.info(f"{colorstr(self.prefix)}: Loaded cached images.")

        if corrupt_example_ids is None:
            corrupt_example_ids = self._get_corrupt_example_ids_from_table(image_paths)
            self._save_cached_example_ids(cache_path, corrupt_example_ids)

        # Filter out corrupt and zero-weight example IDs
        example_ids = self._filter_example_ids(image_paths, corrupt_example_ids)

        im_files, labels = [], []
        for example_id in example_ids:
            im_file = image_paths[example_id]
            im_files.append(im_file)

            row = self.table.table_rows[example_id]
            labels.append(self._get_label_from_row(im_file, row, example_id))

        return im_files, labels

    def _filter_example_ids(self, image_paths: list[str], corrupt_example_ids: list[int]) -> Iterator[int]:
        """Filter example IDs to exclude corrupt and zero-weight images.

        :param image_paths: List of absolute image paths
        :param corrupt_example_ids: List of corrupt example IDs
        :yield: Valid example IDs
        """
        corrupt_set = set(corrupt_example_ids)
        weight_column_name = self.table.weights_column_name

        excluded_count = 0

        for example_id in range(len(image_paths)):
            # Skip corrupt images
            if example_id in corrupt_set:
                continue

            # Skip zero-weight images if exclusion is enabled
            if self._exclude_zero and self.table.table_rows[example_id].get(weight_column_name, 1) == 0:
                excluded_count += 1
                continue

            yield example_id

        if excluded_count > 0:
            percentage_excluded = excluded_count / len(self.table) * 100
            colored_prefix = colorstr(self.prefix + ":")
            LOGGER.info(
                f"{colored_prefix} Excluded {excluded_count} ({percentage_excluded:.2f}% of the table) "
                "zero-weight rows."
            )

    def _get_corrupt_example_ids_from_table(self, image_paths: list[str]) -> list[int]:
        """Get the corrupt example ids from the table by scanning all images.

        :param image_paths: List of absolute image paths
        :return: A list of corrupt example ids
        """
        corrupt_example_ids = []
        verified_count, corrupt_count, msgs = 0, 0, []
        colored_prefix = colorstr(self.prefix + ":")
        desc = f"{colored_prefix} Preparing data from {self.table.url.to_str()}"

        image_iterator = (((im_file, None), "") for im_file in image_paths)

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=image_iterator)
            iterator = enumerate(results)
            pbar = TQDM(iterator, desc=desc, total=len(image_paths))

            for example_id, (_, verified, corrupt, msg) in pbar:
                if verified:
                    verified_count += 1
                elif corrupt:
                    corrupt_example_ids.append(example_id)
                    corrupt_count += 1

                if msg:
                    msgs.append(msg)

                pbar.desc = f"{desc} {verified_count} images, {corrupt_count} corrupt"

            pbar.close()

        if msgs:
            # Only take first 10 messages if there are more
            truncated = len(msgs) > 10
            msgs_to_show = msgs[:10]

            # Create the message string with truncation notice if needed
            msgs_str = "\n".join(msgs_to_show)
            if truncated:
                msgs_str += f"\n... (showing first 10 of {len(msgs)} messages)"

            percentage_corrupt = corrupt_count / len(image_paths) * 100

            verb = "is" if corrupt_count == 1 else "are"
            plural = "s" if corrupt_count != 1 else ""
            LOGGER.warning(
                f"{colored_prefix} There {verb} {corrupt_count} ({percentage_corrupt:.2f}%) corrupt image{plural}:"
                f"\n{msgs_str}"
            )

        return corrupt_example_ids
