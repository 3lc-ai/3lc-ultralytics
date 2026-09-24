from __future__ import annotations

import hashlib
import json
import os
from multiprocessing.pool import ThreadPool
from typing import TYPE_CHECKING, Any

import tlc
from tlc.helpers import ImageHelper
from ultralytics.data.utils import verify_image
from ultralytics.utils import LOGGER, NUM_THREADS, TQDM, colorstr

if TYPE_CHECKING:
    from collections.abc import Iterator


# Responsible for any generic 3LC dataset handling, such as scanning, caching and adding example ids to each sample
# Assume there is an attribute self.table that is a tlc.Table
class TLCDatasetMixin:
    _CACHE_VERSION = 2
    _warned_missing_image_dimensions = False

    _table: tlc.Table | None = None
    _table_url: tlc.Url | None = None

    @property
    def table(self) -> tlc.Table:
        """The `tlc.Table` backing this dataset, restored lazily from its URL after unpickling in a worker."""
        if self._table is None:
            if self._table_url is None:
                msg = "TLCDatasetMixin requires an attribute `table` which is a tlc.Table."
                raise AttributeError(msg)
            self._table = tlc.Table.from_url(self._table_url)
        return self._table

    @table.setter
    def table(self, table: tlc.Table) -> None:
        self._table = table
        self._table_url = table.url if isinstance(table, tlc.Table) else None

    def __getstate__(self) -> dict[str, Any]:
        """Replace Table references with their URLs when pickling to a dataloader worker.

        A Table keeps all of its row data resident in memory, and workers never read it - labels are built once at
        construction. Applies under the `spawn`/`forkserver` worker start methods; `fork` never pickles the dataset.
        """
        state = self.__dict__.copy()
        state["_table"] = None

        if isinstance(state.get("img_path"), tlc.Table) and self._table_url is not None:
            state["img_path"] = self._table_url

        data = state.get("data")
        if isinstance(data, dict):
            state["data"] = {key: value.url if isinstance(value, tlc.Table) else value for key, value in data.items()}

        return state

    def _post_init(self):
        assert isinstance(self._table, tlc.Table), "TLCDatasetMixin requires an attribute `table` which is a tlc.Table."

        self.display_name = self.table.dataset_name

        if len(self.table) == 0:
            msg = f"The Table with URL {self.table.url.to_str()} has no rows, provide a Table populated with data."
            raise ValueError(msg)

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

        if url.scheme not in ("file", "relative"):
            msg = f"Image URL {url.to_str()} is not a local file path, it has scheme {url.scheme}. "
            msg += "Only local image file paths are supported. If your image URLs are not local, first copy "
            msg += "the images to a local directory and use an alias."
            raise ValueError(msg)

        return url.to_absolute(table_url).to_str()

    def _resolve_image_dimensions(self, im_file: str, height: float, width: float) -> tuple[float, float]:
        """Resolve the image ``(height, width)`` for a row.

        Annotations record their own image dimensions. When those are missing or non-positive
        - which makes the labels impossible to decode or normalize - fall back to reading the
        dimensions from the image file instead.

        :param im_file: The absolute path to the image file.
        :param height: The image height recorded in the annotation (``<= 0`` if absent).
        :param width: The image width recorded in the annotation (``<= 0`` if absent).
        :return: The resolved image dimensions as ``(height, width)``.
        """
        if height > 0 and width > 0:
            return height, width

        self._warn_missing_image_dimensions()
        return ImageHelper.get_exif_image_dimensions(im_file)

    def _warn_missing_image_dimensions(self) -> None:
        """Warn, once per dataset, that the Table stores annotations without valid image dimensions."""
        if self._warned_missing_image_dimensions:
            return

        self._warned_missing_image_dimensions = True
        LOGGER.warning(
            f"{colorstr(self.prefix + ':')} Table {self.table.url.to_str()} contains annotations with "
            "non-positive image dimensions. Falling back to reading the dimensions from the image files, "
            "which is slower. This usually means the Table was created without valid image dimensions - "
            "consider re-creating it so the dimensions are stored."
        )

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

    def _load_cached_example_ids(self, cache_url: tlc.Url) -> tuple[list[int], list[int]] | None:
        """Load the cached corrupt and missing example ids from the cache file.

        :param cache_url: The path to the cache file
        :return: `(corrupt_example_ids, missing_example_ids)`, or None if cache is invalid
        """
        try:
            cache_data = json.loads(cache_url.read_text())

            # Check cache version
            if cache_data.get("version") != self._CACHE_VERSION:
                LOGGER.info("Cache version mismatch, regenerating cache.")
                return None

            required_fields = ("corrupt_example_ids", "missing_example_ids")
            if any(field not in cache_data for field in required_fields):
                LOGGER.warning("Cache file is missing image-status fields, regenerating cache.")
                return None

            corrupt_example_ids = cache_data["corrupt_example_ids"]
            missing_example_ids = cache_data["missing_example_ids"]
            if not all(isinstance(example_id, int) for example_id in corrupt_example_ids + missing_example_ids):
                LOGGER.warning("Cache file has invalid image-status fields, regenerating cache.")
                return None
            return corrupt_example_ids, missing_example_ids

        except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            LOGGER.warning(f"Failed to load cache: {e}, regenerating cache.")
            return None

    def _save_cached_example_ids(
        self, cache_url: tlc.Url, corrupt_example_ids: list[int], missing_example_ids: list[int]
    ) -> None:
        """Save the corrupt and missing example ids to the cache file.

        Caching is a pure optimization: the corrupt example ids are already computed in-memory and
        used regardless. If the cache cannot be written (e.g. a parent path component is a file,
        a read-only or full filesystem, or insufficient permissions), warn once and continue
        without a persisted cache rather than aborting dataset construction.

        :param cache_url: The URL to the cache file
        :param corrupt_example_ids: A list of corrupt example ids
        :param missing_example_ids: A list of missing example ids
        """
        content = {
            "version": self._CACHE_VERSION,
            "corrupt_example_ids": corrupt_example_ids,
            "missing_example_ids": missing_example_ids,
        }

        try:
            cache_url.write_text(json.dumps(content, indent=2))
        except OSError as e:
            LOGGER.warning(
                f"{colorstr(self.prefix + ':')} Could not write the images cache to {cache_url.to_str()} "
                f"({e}). Skipping caching and continuing without a persisted cache. This is usually caused by "
                "a stale file where the cache directory should be, or a read-only or full filesystem."
            )

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

        cached_example_ids = self._load_cached_example_ids(cache_path) if cache_path.exists() else None

        if cached_example_ids is not None:
            corrupt_example_ids, missing_example_ids = cached_example_ids
            LOGGER.info(f"{colorstr(self.prefix)}: Loaded cached images.")

            rechecked_example_ids = self._recheck_missing_example_ids(
                image_paths, corrupt_example_ids, missing_example_ids
            )
            if rechecked_example_ids is not None:
                corrupt_example_ids, missing_example_ids = rechecked_example_ids
                self._save_cached_example_ids(cache_path, corrupt_example_ids, missing_example_ids)
        else:
            corrupt_example_ids, missing_example_ids = self._get_invalid_example_ids_from_table(image_paths)
            self._save_cached_example_ids(cache_path, corrupt_example_ids, missing_example_ids)

        if len(missing_example_ids) == len(image_paths):
            msg = (
                f"All images in the Table with URL {self.table.url.to_str()} are missing, can't use it. "
                "This often means that an image URL alias is incorrect or points to an unavailable location."
            )
            raise ValueError(msg)

        if len(corrupt_example_ids) == len(image_paths):
            msg = f"All images in the Table with URL {self.table.url.to_str()} are corrupt, can't use it."
            raise ValueError(msg)

        # Filter out corrupt, missing, and zero-weight example IDs
        invalid_example_ids = corrupt_example_ids + missing_example_ids
        example_ids = list(self._filter_example_ids(image_paths, invalid_example_ids))

        if not example_ids:
            msg = (
                "No valid images found after filtering corrupt, missing, and zero-weight images in the Table with URL "
                f"{self.table.url.to_str()}. Please check the Table and ensure it contains valid images, or provide a "
                "Table with valid images."
            )
            raise ValueError(msg)

        im_files, labels = [], []
        for example_id in example_ids:
            im_file = image_paths[example_id]
            im_files.append(im_file)

            row = self.table.table_rows[example_id]
            labels.append(self._get_label_from_row(im_file, row, example_id))

        return im_files, labels

    def _filter_example_ids(self, image_paths: list[str], invalid_example_ids: list[int]) -> Iterator[int]:
        """Filter example IDs to exclude corrupt, missing, and zero-weight images.

        :param image_paths: List of absolute image paths
        :param invalid_example_ids: List of corrupt or missing example IDs
        :yield: Valid example IDs
        """
        invalid_set = set(invalid_example_ids)
        weight_column_name = self.table.weights_column_name if self._exclude_zero else None

        excluded_count = 0

        for example_id in range(len(image_paths)):
            # Skip corrupt or missing images
            if example_id in invalid_set:
                continue

            # Skip zero-weight images if exclusion is enabled
            if weight_column_name is not None and self.table.table_rows[example_id].get(weight_column_name, 1) == 0:
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

    def _recheck_missing_example_ids(
        self, image_paths: list[str], corrupt_example_ids: list[int], missing_example_ids: list[int]
    ) -> tuple[list[int], list[int]] | None:
        """Verify the images a loaded cache recorded as missing, but which exist now.

        The cache is keyed on image paths only, so an image that appears at an unchanged path, e.g. once an alias'
        mount becomes available, would otherwise stay excluded. Only the cached missing images are checked, which keeps
        the cost proportional to the number of missing images rather than the size of the Table.

        :param image_paths: List of absolute image paths
        :param corrupt_example_ids: Corrupt example ids from the cache
        :param missing_example_ids: Missing example ids from the cache
        :return: Updated `(corrupt_example_ids, missing_example_ids)`, or None if no missing image has appeared
        """
        appeared_example_ids = [
            example_id for example_id in missing_example_ids if os.path.exists(image_paths[example_id])
        ]
        if not appeared_example_ids:
            return None

        LOGGER.info(
            f"{colorstr(self.prefix)}: {len(appeared_example_ids)} previously missing image(s) now exist, "
            "verifying them."
        )
        appeared_paths = [image_paths[example_id] for example_id in appeared_example_ids]
        new_corrupt_indices, new_missing_indices = self._get_invalid_example_ids_from_table(appeared_paths)

        appeared_set = set(appeared_example_ids)
        corrupt_example_ids = sorted(corrupt_example_ids + [appeared_example_ids[i] for i in new_corrupt_indices])
        missing_example_ids = sorted(
            [example_id for example_id in missing_example_ids if example_id not in appeared_set]
            + [appeared_example_ids[i] for i in new_missing_indices]
        )
        return corrupt_example_ids, missing_example_ids

    def _get_invalid_example_ids_from_table(self, image_paths: list[str]) -> tuple[list[int], list[int]]:
        """Get corrupt and missing example ids from the table by scanning all images.

        :param image_paths: List of absolute image paths
        :return: `(corrupt_example_ids, missing_example_ids)`
        """
        corrupt_example_ids, missing_example_ids = [], []
        verified_count, corrupt_count, missing_count = 0, 0, 0
        corrupt_msgs, missing_msgs = [], []
        colored_prefix = colorstr(self.prefix + ":")
        desc = f"{colored_prefix} Preparing data from {self.table.url.to_str()}"

        def verify_image_path(im_file: str) -> tuple[bool, bool, bool, str]:
            try:
                os.stat(im_file)
            except FileNotFoundError:
                return False, False, True, f"{im_file}: missing image file"
            except OSError:
                # Existing-but-unreadable files belong in the corrupt bucket. verify_image supplies its cause.
                pass

            _, verified, corrupt, msg = verify_image(((im_file, None), ""))
            return verified, corrupt, False, msg

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_path, iterable=image_paths)
            iterator = enumerate(results)
            pbar = TQDM(iterator, desc=desc, total=len(image_paths))

            for example_id, (verified, corrupt, missing, msg) in pbar:
                if verified:
                    verified_count += 1
                elif corrupt:
                    corrupt_example_ids.append(example_id)
                    corrupt_count += 1
                elif missing:
                    missing_example_ids.append(example_id)
                    missing_count += 1

                if msg:
                    (missing_msgs if missing else corrupt_msgs).append(msg)

                pbar.desc = f"{desc} {verified_count} images, {missing_count} missing, {corrupt_count} corrupt"

            pbar.close()

        self._log_invalid_images(colored_prefix, "missing", missing_count, len(image_paths), missing_msgs)
        self._log_invalid_images(colored_prefix, "corrupt", corrupt_count, len(image_paths), corrupt_msgs)

        return corrupt_example_ids, missing_example_ids

    @staticmethod
    def _log_invalid_images(prefix: str, status: str, count: int, total: int, msgs: list[str]) -> None:
        """Log one compact warning for missing or corrupt images."""
        if not msgs:
            return

        msgs_to_show = msgs[:10]
        if len(msgs) > len(msgs_to_show):
            msgs_to_show.append(f"... (showing first 10 of {len(msgs)} messages)")

        verb = "is" if count == 1 else "are"
        plural = "" if count == 1 else "s"
        percentage = count / total * 100
        LOGGER.warning(
            f"{prefix} There {verb} {count} ({percentage:.2f}%) {status} image{plural}:\n" + "\n".join(msgs_to_show)
        )
