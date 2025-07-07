#!/usr/bin/env python3
"""
Release script for 3lc-ultralytics package.

This script helps automate the release process by:
1. Updating version in pyproject.toml
2. Updating CHANGELOG.md
3. Creating a git tag
4. Pushing to remote

Usage:
    python scripts/release.py [patch|minor|major|none] [--dry-run]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal


def run_command(cmd: list[str], check: bool = True) -> str:
    """Run a command and return its output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(cmd)}: {e}")
        if check:
            sys.exit(1)
        return ""


def get_current_version() -> str:
    """Get current version from pyproject.toml."""
    command = ["uv", "version", "--output-format", "json"]
    result = run_command(command)
    return json.loads(result)["version"]


def update_version(version_type: Literal["major", "minor", "patch", "none"], dry_run: bool = False) -> str:
    """Update version in pyproject.toml and return new version.

    :param version_type: The type of version bump to apply.
    :param dry_run: If True, do not actually update the version.
    :returns: The new version.
    """

    if version_type == "none":
        return get_current_version()

    command = ["uv", "version", "--bump", version_type, "--output-format", "json"]
    if dry_run:
        command.append("--dry-run")
    result = run_command(command)
    return json.loads(result)["version"]


def update_changelog(new_version: str, dry_run: bool = False) -> str:
    """Update CHANGELOG.md to move unreleased changes to the new version.

    :param new_version: The version being released
    :param dry_run: If True, return the new content without writing to file
    :returns: The updated changelog content
    """
    changelog_path = Path("CHANGELOG.md")
    content = changelog_path.read_text()

    # Replace the Unreleased section with new Unreleased at top + versioned section below
    # This regex captures the entire Unreleased section and replaces it
    # Handles both "## [Unreleased]" and "## [Unreleased] (X.Y.Z)" formats
    new_content = re.sub(
        r'## \[Unreleased\](?: \([^)]+\))?\n(.*?)(?=\n## \[|$)',
        f'## [Unreleased]\n\n## [{new_version}] - {get_release_date()}\n\\1',
        content,
        flags=re.DOTALL
    )

    if not dry_run:
        changelog_path.write_text(new_content)

    return new_content


def get_release_date() -> str:
    """Get current date in YYYY-MM-DD format."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")


def create_git_tag(version: str, dry_run: bool = False) -> None:
    """Create and push git tag."""
    tag_name = f"v{version}"

    if dry_run:
        print(f"[DRY RUN] Would create tag: {tag_name}")
        return

    # Check if we're on develop branch
    current_branch = run_command(["git", "branch", "--show-current"])
    if current_branch != "develop":
        print(f"Warning: Not on develop branch (currently on {current_branch})")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != "y":
            sys.exit(1)

    # Check for uncommitted changes
    status = run_command(["git", "status", "--porcelain"])
    if status:
        print("Warning: There are uncommitted changes:")
        print(status)
        response = input("Continue anyway? (y/N): ")
        if response.lower() != "y":
            sys.exit(1)

    # Create tag
    run_command(["git", "tag", tag_name])
    print(f"Created tag: {tag_name}")

    # Push tag
    run_command(["git", "push", "origin", tag_name])
    print(f"Pushed tag: {tag_name}")


def main():
    parser = argparse.ArgumentParser(description="Release script for 3lc-ultralytics")
    parser.add_argument("version_type", choices=["patch", "minor", "major", "none"], help="Type of version bump")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")

    args = parser.parse_args()

    print(f"Current version: {get_current_version()}")

    if args.dry_run:
        print("[DRY RUN MODE]")

    # Update version
    new_version = update_version(args.version_type, dry_run=args.dry_run)
    print(f"{'[DRY RUN] ' if args.dry_run else ''}New version: {new_version}")

    # Update changelog
    if args.dry_run:
        modified_content = update_changelog(new_version, dry_run=True)
        print(f"\n[DRY RUN] CHANGELOG.md would be updated to:\n{'-' * 80}")
        print(modified_content)
        print(f"{'-' * 80}")
    else:
        update_changelog(new_version)
        print("Updated CHANGELOG.md")

    if not args.dry_run:
        # Commit changes
        run_command(["git", "add", "pyproject.toml", "CHANGELOG.md"])
        run_command(["git", "commit", "-m", f"Bump version to {new_version}"])
        print("Committed version changes")

        # Push changes to develop
        run_command(["git", "push", "origin", "develop"])
        print("Pushed changes to develop branch")

        # Create and push tag
        create_git_tag(new_version, dry_run=args.dry_run)

        print(f"\n🎉 Release {new_version} is ready!")
        print("The GitHub Actions workflow will automatically:")
        print("1. Run tests")
        print("2. Build the package")
        print("3. Publish to PyPI")
        print("4. Create a GitHub release")
    else:
        print("\n[DRY RUN] Would have:")
        print("1. Added and committed these changes to develop branch")
        print("2. Pushed these changes to develop branch")
        print("3. Created and pushed tag")
        print(
            "4. Triggered GitHub Actions workflow to test, build, publish and release the package as GitHub Release "
            "and on PyPI"
        )


if __name__ == "__main__":
    main()
