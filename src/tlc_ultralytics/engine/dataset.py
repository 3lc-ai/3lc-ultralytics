from __future__ import annotations

import hashlib
import json
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
    _warned_missing_image_dimensions = False

    # Backing state for the `table` property. Class-level defaults so `hasattr(self, "table")` is answerable
    # before any subclass has assigned a Table.
    _table: tlc.Table | None = None
    _table_url: tlc.Url | None = None

    @property
    def table(self) -> tlc.Table:
        """The `tlc.Table` backing this dataset, restored from its URL if it was dropped when the dataset was
        pickled to a dataloader worker. See `__getstate__` for why the Table does not cross that boundary.

        :return: The Table this dataset was built from.
        :raises AttributeError: If no Table has been assigned to this dataset.
        """
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
        """Drop every reference to the `tlc.Table` from the state pickled to dataloader workers.

        Workers only ever read `self.labels` and `self.im_files` - the per-item path never touches the Table, which
        is fully consumed at construction time by `_get_rows_from_table`. A Table however holds all of its row data
        resident as a `pyarrow.Table`, and `tlc.Table.__getstate__` pickles that data as-is, so any surviving
        reference ships the entire dataset's annotations to every worker. On platforms where dataloader workers are
        spawned rather than forked (macOS, Windows, and `spawn` start methods generally) that is `workers` extra
        copies of data nothing reads. All references have to go, not just `self.table`: pickle memoizes, so a single
        surviving one costs the full Table.

        The references are `self.table`, `self.img_path` (Ultralytics stores the first constructor argument, which
        is the Table for our datasets) and the `train`/`val` entries of `self.data`. Each is replaced by the
        corresponding `tlc.Url`; `self.table` is restored lazily from that URL should anything ask for it.

        :return: The dataset state to pickle, with Tables replaced by their URLs.
        """
        state = self.__dict__.copy()
        state["_table"] = None

        if isinstance(state.get("img_path"), tlc.Table) and self._table_url is not None:
            state["img_path"] = self._table_url.to_str()

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

    def _load_cached_example_ids(self, cache_url: tlc.Url) -> list[int] | None:
        """Load the cached corrupt example ids from the cache file.

        :param cache_url: The path to the cache file
        :return: A list of corrupt example ids, or None if cache is invalid
        """
        try:
            cache_data = json.loads(cache_url.read_text())

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

        except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            LOGGER.warning(f"Failed to load cache: {e}, regenerating cache.")
            return None

    def _save_cached_example_ids(self, cache_url: tlc.Url, corrupt_example_ids: list[int]):
        """Save the corrupt example ids to the cache file.

        Caching is a pure optimization: the corrupt example ids are already computed in-memory and
        used regardless. If the cache cannot be written (e.g. a parent path component is a file,
        a read-only or full filesystem, or insufficient permissions), warn once and continue
        without a persisted cache rather than aborting dataset construction.

        :param cache_url: The URL to the cache file
        :param corrupt_example_ids: A list of corrupt example ids
        """
        content = {
            "version": 1,
            "corrupt_example_ids": corrupt_example_ids,
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

        corrupt_example_ids = self._load_cached_example_ids(cache_path) if cache_path.exists() else None

        if corrupt_example_ids is not None:
            LOGGER.info(f"{colorstr(self.prefix)}: Loaded cached images.")

        if corrupt_example_ids is None:
            corrupt_example_ids = self._get_corrupt_example_ids_from_table(image_paths)
            self._save_cached_example_ids(cache_path, corrupt_example_ids)

        if len(corrupt_example_ids) == len(image_paths):
            msg = f"All images in the Table with URL {self.table.url.to_str()} are corrupt, can't use it."
            raise ValueError(msg)

        # Filter out corrupt and zero-weight example IDs
        example_ids = list(self._filter_example_ids(image_paths, corrupt_example_ids))

        if not example_ids:
            msg = (
                "No valid images found after filtering corrupt and zero-weight images in the Table with URL "
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

    def _filter_example_ids(self, image_paths: list[str], corrupt_example_ids: list[int]) -> Iterator[int]:
        """Filter example IDs to exclude corrupt and zero-weight images.

        :param image_paths: List of absolute image paths
        :param corrupt_example_ids: List of corrupt example IDs
        :yield: Valid example IDs
        """
        corrupt_set = set(corrupt_example_ids)

        # A table without a weights column has no zero-weight rows to exclude.
        weight_column_name = self.table.weights_column_name if self._exclude_zero else None

        excluded_count = 0

        for example_id in range(len(image_paths)):
            # Skip corrupt images
            if example_id in corrupt_set:
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
