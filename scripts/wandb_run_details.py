# focused_wandb_dump.py
from wandb import Api
import json
import math
import numpy as np

api = Api()
RUN_PATH = "divyanshukla/DinoGemmaCaptionerQFormer/ewy64sj9"  # use entity/project/run_id

def make_json_serializable(obj):
    """Recursively convert to JSON-safe Python types."""
    # basic primitives
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj

    # numpy scalars/arrays
    if isinstance(obj, (np.generic, np.number)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # dict-like
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(v) for v in obj]

    # Try dict(...) for W&B SDK objects like HTTPSummary / SummarySubDict
    try:
        maybe = dict(obj)
        if isinstance(maybe, dict):
            return {str(k): make_json_serializable(v) for k, v in maybe.items()}
    except Exception:
        pass

    # Try common conversion methods
    for attr in ("to_dict", "tolist", "item", "value"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return make_json_serializable(fn())
            except Exception:
                pass

    # Fallback to string
    try:
        return str(obj)
    except Exception:
        return None

def main():
    run = api.run(RUN_PATH)
    # convert summary and config
    summary_safe = make_json_serializable(dict(run.summary))
    config_safe = make_json_serializable(dict(run.config))
    attrs_safe = make_json_serializable(run._attrs)

    # write JSON files
    with open("run_summary.json", "w", encoding="utf-8") as jf:
        json.dump(summary_safe, jf, indent=2, ensure_ascii=False)

    with open("run_config.json", "w", encoding="utf-8") as jf:
        json.dump(config_safe, jf, indent=2, ensure_ascii=False)

    # human-readable combined text
    with open("run_dump.txt", "w", encoding="utf-8") as f:
        f.write("=== RUN PATH ===\n" + RUN_PATH + "\n\n")
        f.write("=== CONFIG ===\n")
        f.write(json.dumps(config_safe, indent=2))
        f.write("\n\n=== SUMMARY ===\n")
        f.write(json.dumps(summary_safe, indent=2))
        f.write("\n\n=== ATTRS ===\n")
        f.write(json.dumps(attrs_safe, indent=2))

    print("Wrote run_summary.json, run_config.json and run_dump.txt")

if __name__ == "__main__":
    main()
