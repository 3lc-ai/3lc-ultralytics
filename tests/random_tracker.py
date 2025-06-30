#!/usr/bin/env python3
# =============================================================================
# <copyright>
# Copyright (c) 2025 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================

import random
import traceback
from collections import defaultdict
from typing import Any, Union, cast

import numpy as np
import torch

from torch import _C

# Store the original random functions
original_random = random.random
original_randint = random.randint
original_choice = random.choice
original_shuffle = random.shuffle
original_sample = random.sample
original_np_random = np.random.random
original_np_randint = np.random.randint
original_np_choice = np.random.choice
original_np_shuffle = np.random.shuffle
original_torch_manual_seed = torch.manual_seed

# Dictionary to store call counts and call stacks
call_counts: defaultdict[str, int] = defaultdict(int)
call_stacks: defaultdict[str, list[str]] = defaultdict(list)


def reset_tracking() -> None:
    """Reset the tracking counters."""
    call_counts.clear()
    call_stacks.clear()


def get_tracking_info() -> dict[str, dict[str, Union[int, list[str]]]]:
    """Get the current tracking information."""
    return {"call_counts": dict(call_counts), "call_stacks": dict(call_stacks)}


def get_formatted_call_stack(stack: list[traceback.FrameSummary], max_depth: int = 4) -> str:
    """Get a formatted call stack with multiple levels of parent calls.

    Args:
        stack: The call stack from traceback.extract_stack()
        max_depth: Maximum number of parent calls to include

    Returns:
        A formatted string with indented call stack information
    """
    # Filter out frames from this file
    relevant_frames = [frame for frame in stack if frame.filename != __file__]

    # Take the most recent frames up to max_depth
    relevant_frames = relevant_frames[-max_depth:] if len(relevant_frames) > max_depth else relevant_frames

    # Format the call stack with indentation
    formatted_stack = []
    for i, frame in enumerate(relevant_frames):
        indent = "  " * i
        formatted_stack.append(f"{indent}{frame.filename}:{frame.lineno}")

    return "\n".join(formatted_stack)


# Wrapper functions for random module
def tracked_random(*args: Any, **kwargs: Any) -> float:
    call_counts["random.random"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["random.random"].append(formatted_stack)
    return original_random(*args, **kwargs)


def tracked_randint(*args: Any, **kwargs: Any) -> int:
    call_counts["random.randint"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["random.randint"].append(formatted_stack)
    return original_randint(*args, **kwargs)


def tracked_choice(*args: Any, **kwargs: Any) -> Any:
    call_counts["random.choice"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["random.choice"].append(formatted_stack)
    return original_choice(*args, **kwargs)


def tracked_shuffle(*args: Any, **kwargs: Any) -> None:
    call_counts["random.shuffle"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["random.shuffle"].append(formatted_stack)
    return original_shuffle(*args, **kwargs)


def tracked_sample(*args: Any, **kwargs: Any) -> list[Any]:
    call_counts["random.sample"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["random.sample"].append(formatted_stack)
    return original_sample(*args, **kwargs)


# Wrapper functions for numpy.random
def tracked_np_random(*args: Any, **kwargs: Any) -> np.ndarray:
    call_counts["np.random.random"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["np.random.random"].append(formatted_stack)
    return original_np_random(*args, **kwargs)


def tracked_np_randint(*args: Any, **kwargs: Any) -> Union[int, np.ndarray]:
    call_counts["np.random.randint"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["np.random.randint"].append(formatted_stack)
    return original_np_randint(*args, **kwargs)


def tracked_np_choice(*args: Any, **kwargs: Any) -> Union[Any, np.ndarray]:
    call_counts["np.random.choice"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["np.random.choice"].append(formatted_stack)
    return original_np_choice(*args, **kwargs)


def tracked_np_shuffle(*args: Any, **kwargs: Any) -> None:
    call_counts["np.random.shuffle"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["np.random.shuffle"].append(formatted_stack)
    return original_np_shuffle(*args, **kwargs)


# Wrapper for torch.manual_seed
def tracked_torch_manual_seed(seed: Any) -> _C.Generator:
    call_counts["torch.manual_seed"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["torch.manual_seed"].append(formatted_stack)
    return original_torch_manual_seed(seed)


def enable_tracking() -> None:
    """Enable tracking of random calls."""
    random.random = tracked_random
    random.randint = tracked_randint
    random.choice = tracked_choice
    random.shuffle = tracked_shuffle
    random.sample = tracked_sample
    np.random.random = cast(Any, tracked_np_random)
    np.random.randint = cast(Any, tracked_np_randint)
    np.random.choice = cast(Any, tracked_np_choice)
    np.random.shuffle = tracked_np_shuffle
    torch.manual_seed = tracked_torch_manual_seed


def disable_tracking() -> None:
    """Disable tracking and restore original functions."""
    random.random = original_random
    random.randint = original_randint
    random.choice = original_choice
    random.shuffle = original_shuffle
    random.sample = original_sample
    np.random.random = original_np_random
    np.random.randint = original_np_randint
    np.random.choice = original_np_choice
    np.random.shuffle = original_np_shuffle
    torch.manual_seed = original_torch_manual_seed


def print_call_summary(title: str, call_counts: dict[str, int]) -> None:
    """Print a summary of random function calls.

    Args:
        title: Title for the summary section
        call_counts: Dictionary of function names to call counts
    """
    print(f"\n{title}:")
    for func, count in call_counts.items():
        print(f"{func}: {count} calls")


def find_different_call_counts(
    call_counts_1: dict[str, int], call_counts_2: dict[str, int]
) -> dict[str, tuple[int, int]]:
    """Find functions with different call counts between two sets.

    Args:
        call_counts_1: First set of call counts
        call_counts_2: Second set of call counts

    Returns:
        Dictionary mapping function names to tuples of (count_1, count_2)
    """
    different_counts: dict[str, tuple[int, int]] = {}
    all_funcs = set(call_counts_1.keys()) | set(call_counts_2.keys())

    for func in all_funcs:
        count_1 = call_counts_1.get(func, 0)
        count_2 = call_counts_2.get(func, 0)

        if count_1 != count_2:
            different_counts[func] = (count_1, count_2)

    return different_counts


def print_call_locations(
    func: str, call_stacks_1: dict[str, list[str]], call_stacks_2: dict[str, list[str]], title_1: str, title_2: str
) -> None:
    """Print call locations for a function from two different sources.

    Args:
        func: Function name
        call_stacks_1: Call stacks from first source
        call_stacks_2: Call stacks from second source
        title_1: Title for first source
        title_2: Title for second source
    """
    if func in call_stacks_1:
        print(f"{title_1} call locations for {func}:")
        for location in set(call_stacks_1[func]):
            print(f"  {location}")

    if func in call_stacks_2:
        print(f"{title_2} call locations for {func}:")
        for location in set(call_stacks_2[func]):
            print(f"  {location}")


def add_tracking_info_to_rows(
    rows: list[dict[str, Any]], call_counts: dict[str, int], different_call_counts: dict[str, tuple[int, int]]
) -> None:
    """Add random tracking information to dataset rows.

    Args:
        rows: List of dataset rows
        call_counts: Dictionary of function names to call counts
        different_call_counts: Dictionary of functions with different call counts
    """
    for row in rows:
        row["random_tracking_info"] = {
            "call_counts": call_counts,
            "different_call_counts": different_call_counts,
        }


def print_random_info(
    mode: str,
    random_info_3lc: dict[str, Any],
    random_info_ultralytics: dict[str, Any],
    rows_3lc: list[dict[str, Any]],
    rows_ultralytics: list[dict[str, Any]],
) -> None:
    """Print information about random function calls and add tracking info to rows.

    Args:
        mode: Dataset mode (train or val)
        random_info_3lc: Random tracking info from 3LC
        random_info_ultralytics: Random tracking info from Ultralytics
        rows_3lc: Dataset rows from 3LC
        rows_ultralytics: Dataset rows from Ultralytics
    """
    print(f"\n--- Random call information for mode: {mode} ---")

    # Print call summaries
    print_call_summary("3LC random calls", random_info_3lc["call_counts"])
    print_call_summary("Ultralytics random calls", random_info_ultralytics["call_counts"])

    # Find and print differences in call counts
    different_call_counts = find_different_call_counts(
        random_info_3lc["call_counts"], random_info_ultralytics["call_counts"]
    )

    for func, (count_3lc, count_ultralytics) in different_call_counts.items():
        print(f"\nDifferent call counts for {func}: 3LC={count_3lc}, Ultralytics={count_ultralytics}")
        print_call_locations(
            func, random_info_3lc["call_stacks"], random_info_ultralytics["call_stacks"], "3LC", "Ultralytics"
        )

    # Add tracking info to rows
    add_tracking_info_to_rows(rows_3lc, random_info_3lc["call_counts"], different_call_counts)
    add_tracking_info_to_rows(rows_ultralytics, random_info_ultralytics["call_counts"], different_call_counts)
