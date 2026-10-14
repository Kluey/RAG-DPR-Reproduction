import json
from pathlib import Path

from ragdpr.config import RESULTS_FILE


def load_results(path: str = RESULTS_FILE) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_result(keys: list[str], value, path: str = RESULTS_FILE):
    results = load_results(path)
    node = results
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {'.'.join(keys)} to {path}")
