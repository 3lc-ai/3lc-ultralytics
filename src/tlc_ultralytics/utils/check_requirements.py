from __future__ import annotations

import importlib

from packaging.specifiers import SpecifierSet

from tlc_ultralytics.constants import REQUIREMENTS_TO_CHECK


def check_requirements(requirements_to_check: list[tuple[str, str]] = REQUIREMENTS_TO_CHECK) -> None:
    """Check the versions of the required packages matches the version specifiers in pyproject.toml.

    :param requirements_to_check: List of tuples of (package name, import name) of packages to check.
    :raises ImportError: If the requirements are not met.
    """
    tlc_ultralytics_requirements = importlib.metadata.metadata("3lc-ultralytics").json["requires_dist"]

    for package_name, import_name in requirements_to_check:
        try:
            importlib.import_module(import_name)
        except ImportError:
            raise ImportError(f"Failed to import '{import_name}', please install it.") from None

        # Check the version of the package and match it against the version set specifier
        installed_version = importlib.metadata.version(package_name)
        required_version_specifier = next(
            version_specifier.replace(package_name, "")
            for version_specifier in tlc_ultralytics_requirements
            if version_specifier.startswith(package_name)
        )

        if required_version_specifier is None:
            raise ValueError(f"No version specifier found for '{package_name}'.")

        if installed_version not in SpecifierSet(required_version_specifier):
            raise ImportError(
                f"The installed version of '{package_name}' ({installed_version}) is outside the required version "
                f"range {required_version_specifier}. Please install a compatible version."
            )
