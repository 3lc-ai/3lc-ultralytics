import random
import traceback
from typing import Callable


class RandomTracker:
    """Track calls to the global random number generator with call site information."""

    def __init__(self):
        self.call_count = 0
        self.call_details: list[dict] = []
        self.original_random = random.random
        self.original_randint = random.randint
        self.original_choice = random.choice
        self.original_shuffle = random.shuffle
        self.original_seed = random.seed
        self.original_getstate = random.getstate
        self.original_setstate = random.setstate
        self.original_uniform = random.uniform
        self.original_randrange = random.randrange
        self.original_sample = random.sample
        self.original_choices = random.choices

    def _get_call_info(self, func_name: str) -> dict:
        """Get information about the current call site."""
        try:
            stack = traceback.extract_stack()
            # Find first frame not in this module
            for frame in reversed(stack):
                if "random_tracker" not in frame.filename:
                    return {
                        "function": func_name,
                        "module": frame.filename,
                        "line": frame.lineno,
                        "function_name": frame.name,
                        "line_content": frame.line.strip() if frame.line else "",
                        "full_path": frame.filename,
                    }
            return {
                "function": func_name,
                "module": "unknown",
                "line": 0,
                "function_name": "unknown",
                "line_content": "",
                "full_path": "unknown",
            }
        except:  # noqa: E722
            return {
                "function": func_name,
                "module": "error",
                "line": 0,
                "function_name": "error",
                "line_content": "",
                "full_path": "error",
            }

    def _count_call(self, func: Callable, func_name: str) -> Callable:
        """Wrap a function to count calls and capture call site info."""

        def wrapper(*args, **kwargs):
            self.call_count += 1
            call_info = self._get_call_info(func_name)
            self.call_details.append(call_info)
            return func(*args, **kwargs)

        return wrapper

    def start_tracking(self):
        """Start tracking random calls by replacing global random functions."""
        random.random = self._count_call(self.original_random, "random")
        random.randint = self._count_call(self.original_randint, "randint")
        random.choice = self._count_call(self.original_choice, "choice")
        random.shuffle = self._count_call(self.original_shuffle, "shuffle")
        random.seed = self._count_call(self.original_seed, "seed")
        random.getstate = self._count_call(self.original_getstate, "getstate")
        random.setstate = self._count_call(self.original_setstate, "setstate")
        random.uniform = self._count_call(self.original_uniform, "uniform")
        random.randrange = self._count_call(self.original_randrange, "randrange")
        random.sample = self._count_call(self.original_sample, "sample")
        random.choices = self._count_call(self.original_choices, "choices")

    def stop_tracking(self):
        """Stop tracking and restore original random functions."""
        random.random = self.original_random
        random.randint = self.original_randint
        random.choice = self.original_choice
        random.shuffle = self.original_shuffle
        random.seed = self.original_seed
        random.getstate = self.original_getstate
        random.setstate = self.original_setstate
        random.uniform = self.original_uniform
        random.randrange = self.original_randrange
        random.sample = self.original_sample
        random.choices = self.original_choices

    def get_count(self) -> int:
        """Get the current call count."""
        return self.call_count

    def reset_count(self):
        """Reset the call count and details."""
        self.call_count = 0
        self.call_details.clear()

    def get_call_summary(self) -> dict[str, int]:
        """Get a summary of calls by module."""
        summary = {}
        for call in self.call_details:
            module = call["module"]
            summary[module] = summary.get(module, 0) + 1
        return summary

    def get_detailed_calls(self) -> list[dict]:
        """Get detailed information about all calls."""
        return self.call_details.copy()

    def get_calls_by_module(self, module_name: str) -> list[dict]:
        """Get all calls from a specific module."""
        return [call for call in self.call_details if module_name in call["module"]]


# Global tracker instance
_tracker = RandomTracker()


def start_tracking():
    """Start tracking random calls globally."""
    _tracker.start_tracking()


def stop_tracking():
    """Stop tracking random calls globally."""
    _tracker.stop_tracking()


def get_random_call_count() -> int:
    """Get the total number of random calls made."""
    return _tracker.get_count()


def reset_random_call_count():
    """Reset the random call counter."""
    _tracker.reset_count()


def print_random_call_count(script_name: str):
    """Print the random call count and detailed breakdown for a script."""
    count = get_random_call_count()
    print(f"[{script_name}] Total random calls: {count}")

    if count > 0:
        summary = _tracker.get_call_summary()
        print(f"[{script_name}] Call breakdown by module:")
        for module, module_count in sorted(summary.items()):
            short_module = module.split("/")[-1] if "/" in module else module
            print(f"  {short_module}: {module_count} calls")

        # Show first few detailed calls
        detailed_calls = _tracker.get_detailed_calls()
        if detailed_calls:
            print(f"[{script_name}] First 5 detailed calls:")
            for i, call in enumerate(detailed_calls[:5]):
                short_module = call["module"].split("/")[-1] if "/" in call["module"] else call["module"]
                print(f"  {i + 1}. {call['function']} at {short_module}:{call['line']} ({call['function_name']})")
                if call["line_content"]:
                    print(f"     Line: {call['line_content']}")

    return count


def get_call_details() -> list[dict]:
    """Get detailed information about all random calls."""
    return _tracker.get_detailed_calls()


def compare_calls(calls_3lc: list[dict], calls_ultralytics: list[dict]) -> dict:
    """Compare calls between 3LC and ultralytics runs and return differences."""

    # Create call signatures for comparison
    def create_call_signature(call: dict) -> str:
        return f"{call['full_path']}:{call['line']}:{call['function']}"

    calls_3lc_sigs = {create_call_signature(call) for call in calls_3lc}
    calls_ultralytics_sigs = {create_call_signature(call) for call in calls_ultralytics}

    # Find differences
    only_in_3lc = calls_3lc_sigs - calls_ultralytics_sigs
    calls_ultralytics_sigs - calls_3lc_sigs

    # Get detailed info for calls only in 3LC
    calls_only_in_3lc = []
    for call in calls_3lc:
        sig = create_call_signature(call)
        if sig in only_in_3lc:
            calls_only_in_3lc.append(call)

    return {
        "only_in_3lc": calls_only_in_3lc,
        "only_in_ultralytics": list(calls_ultralytics_sigs - calls_3lc_sigs),
        "total_3lc": len(calls_3lc),
        "total_ultralytics": len(calls_ultralytics),
        "common": len(calls_3lc_sigs & calls_ultralytics_sigs),
    }


def print_call_comparison(comparison: dict):
    """Print a detailed comparison of random calls."""
    print("\nCall comparison summary:")
    print(f"  3LC total calls: {comparison['total_3lc']}")
    print(f"  Ultralytics total calls: {comparison['total_ultralytics']}")
    print(f"  Common calls: {comparison['common']}")
    print(f"  Calls only in 3LC: {len(comparison['only_in_3lc'])}")
    print(f"  Calls only in Ultralytics: {len(comparison['only_in_ultralytics'])}")

    if comparison["only_in_3lc"]:
        print("\nCalls only in 3LC:")
        for i, call in enumerate(comparison["only_in_3lc"], 1):
            short_module = call["module"].split("/")[-1] if "/" in call["module"] else call["module"]
            print(f"  {i}. {call['function']} at {short_module}:{call['line']} ({call['function_name']})")
            if call["line_content"]:
                print(f"     Line: {call['line_content']}")

    if comparison["only_in_ultralytics"]:
        print("\nCalls only in Ultralytics:")
        for i, call_sig in enumerate(comparison["only_in_ultralytics"], 1):
            print(f"  {i}. {call_sig}")
