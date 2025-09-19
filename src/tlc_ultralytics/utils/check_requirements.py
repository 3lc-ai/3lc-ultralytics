from __future__ import annotations

import importlib

from packaging.specifiers import SpecifierSet
from ultralytics.utils import LOGGER

from tlc_ultralytics.constants import REQUIREMENTS_TO_CHECK


def check_requirements(requirements_to_check: list[tuple[str, str]] = REQUIREMENTS_TO_CHECK) -> None:
    """Check the versions of the required packages are installed, and warn if the versions are not
    known to be compatible.

    :param requirements_to_check: List of tuples of (package name, import name) of packages to check.
    :raises ImportError: If the requirements are not installed.
    """
    tlc_ultralytics_requirements = importlib.metadata.metadata("3lc-ultralytics").json["requires_dist"]

    for package_name, import_name in requirements_to_check:
        try:
            importlib.import_module(import_name)
        except ImportError:
            msg = (
                f"Failed to import '{import_name}', which is a required dependency of 3lc-ultralytics. "
                f"Please install it with `pip install {package_name}` or equivalent."
            )
            raise ImportError(msg) from None

        # Check the version of the package and match it against the version set specifier
        installed_version = importlib.metadata.version(package_name)
        required_version_specifier = next(
            version_specifier.replace(package_name, "")
            for version_specifier in tlc_ultralytics_requirements
            if version_specifier.startswith(package_name)
        )

        if required_version_specifier is None:
            LOGGER.info(f"No version specifier found for '{package_name}', skipping version check.")

        if installed_version not in SpecifierSet(required_version_specifier):
            LOGGER.warning(
                f"The installed version of '{package_name}' ({installed_version}) is outside the required version "
                f"range {required_version_specifier}. This may cause compatibility issues, please use a compatible "
                "version if issues are encountered."
            )
