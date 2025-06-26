import random
import traceback
import numpy as np
import torch
from collections import defaultdict

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
call_counts = defaultdict(int)
call_stacks = defaultdict(list)


def reset_tracking():
    """Reset the tracking counters."""
    call_counts.clear()
    call_stacks.clear()


def get_tracking_info():
    """Get the current tracking information."""
    return {"call_counts": dict(call_counts), "call_stacks": dict(call_stacks)}


def get_formatted_call_stack(stack, max_depth=4):
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
def tracked_random(*args, **kwargs):
    call_counts["random.random"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["random.random"].append(formatted_stack)
    return original_random(*args, **kwargs)


def tracked_randint(*args, **kwargs):
    call_counts["random.randint"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["random.randint"].append(formatted_stack)
    return original_randint(*args, **kwargs)


def tracked_choice(*args, **kwargs):
    call_counts["random.choice"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["random.choice"].append(formatted_stack)
    return original_choice(*args, **kwargs)


def tracked_shuffle(*args, **kwargs):
    call_counts["random.shuffle"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["random.shuffle"].append(formatted_stack)
    return original_shuffle(*args, **kwargs)


def tracked_sample(*args, **kwargs):
    call_counts["random.sample"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["random.sample"].append(formatted_stack)
    return original_sample(*args, **kwargs)


# Wrapper functions for numpy.random
def tracked_np_random(*args, **kwargs):
    call_counts["np.random.random"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["np.random.random"].append(formatted_stack)
    return original_np_random(*args, **kwargs)


def tracked_np_randint(*args, **kwargs):
    call_counts["np.random.randint"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["np.random.randint"].append(formatted_stack)
    return original_np_randint(*args, **kwargs)


def tracked_np_choice(*args, **kwargs):
    call_counts["np.random.choice"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["np.random.choice"].append(formatted_stack)
    return original_np_choice(*args, **kwargs)


def tracked_np_shuffle(*args, **kwargs):
    call_counts["np.random.shuffle"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["np.random.shuffle"].append(formatted_stack)
    return original_np_shuffle(*args, **kwargs)


# Wrapper for torch.manual_seed
def tracked_torch_manual_seed(*args, **kwargs):
    call_counts["torch.manual_seed"] += 1
    stack = traceback.extract_stack()
    formatted_stack = get_formatted_call_stack(stack)
    if formatted_stack:
        call_stacks["torch.manual_seed"].append(formatted_stack)
    return original_torch_manual_seed(*args, **kwargs)


def enable_tracking():
    """Enable tracking of random calls."""
    random.random = tracked_random
    random.randint = tracked_randint
    random.choice = tracked_choice
    random.shuffle = tracked_shuffle
    random.sample = tracked_sample
    np.random.random = tracked_np_random
    np.random.randint = tracked_np_randint
    np.random.choice = tracked_np_choice
    np.random.shuffle = tracked_np_shuffle
    torch.manual_seed = tracked_torch_manual_seed


def disable_tracking():
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
