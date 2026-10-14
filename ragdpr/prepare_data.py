import argparse
import json
from pathlib import Path

from datasets import load_dataset

from ragdpr.config import DATA_DIR, DATASET_NAME, N_DEV, N_HARD_NEGATIVES, N_TRAIN


def convert_example(row: dict, n_hard_negatives: int) -> dict:
    return {
        "question": row["query"],
        "answers": list(row["answers"]),
        "positive": to_passage(row["positive_passages"][0]),
        "hard_negatives": [to_passage(p) for p in row["negative_passages"][:n_hard_negatives]],
    }


def to_passage(p: dict) -> dict:
    return {"id": str(p["docid"]), "title": p["title"], "text": p["text"]}


def collect_passages(rows: list[dict]) -> dict[str, dict]:
    passages = {}
    for row in rows:
        for p in row["positive_passages"] + row["negative_passages"]:
            passage = to_passage(p)
            passages.setdefault(passage["id"], passage)
    return passages


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=N_TRAIN)
    parser.add_argument("--n_dev", type=int, default=N_DEV)
    args = parser.parse_args()

    out_dir = Path(DATA_DIR)
    dataset = load_dataset(DATASET_NAME, revision="refs/convert/parquet")

    splits = {"train": ("train", args.n_train), "dev": ("validation", args.n_dev)}
    all_rows: list[dict] = []
    for split, (hf_split, n) in splits.items():
        rows: list[dict] = []
        for row in dataset[hf_split]:
            if len(rows) >= n:
                break
            if row["positive_passages"]:  # type: ignore[index]
                rows.append(dict(row))  
        write_jsonl(out_dir / f"{split}.jsonl", [convert_example(r, N_HARD_NEGATIVES) for r in rows])
        print(f"{split}: {len(rows)} examples")
        all_rows.extend(rows)

    passages = collect_passages(all_rows)
    write_jsonl(out_dir / "passages.jsonl", list(passages.values()))
    print(f"passages: {len(passages)}")


if __name__ == "__main__":
    main()
