from collections import Counter
import json

with open("tmp/3lc_random_calls.json") as f:
    calls_3lc = json.load(f)
with open("tmp/ultralytics_random_calls.json") as f:
    calls_ultralytics = json.load(f)


def filter_augment_calls(calls):
    return [
        (c["line"], c["function"], c["function_name"], c["line_content"]) for c in calls if "augment.py" in c["module"]
    ]


calls_3lc_aug = filter_augment_calls(calls_3lc)
calls_ultra_aug = filter_augment_calls(calls_ultralytics)

counter_3lc = Counter(calls_3lc_aug)
counter_ultra = Counter(calls_ultra_aug)

print("Line/function/line_content : 3LC count vs Ultralytics count")
for key in sorted(set(counter_3lc) | set(counter_ultra)):
    c3 = counter_3lc.get(key, 0)
    cu = counter_ultra.get(key, 0)
    if c3 != cu:
        print(f"{key} : 3LC={c3}  Ultralytics={cu}  DIFF={c3 - cu:+d}")
