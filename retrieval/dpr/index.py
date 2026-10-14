import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer

from config import DEV_FILE, DEVICE, DPR_DIR, ENCODE_BATCH_SIZE, INDEX_PATH, MAX_P_LEN, MAX_Q_LEN, PASSAGES_FILE
from metrics import has_answer
from retrieval.dpr.model import Encoder
from retrieval.dpr.train import format_passage


def load_passages(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_encoder(which: str) -> Encoder:
    return Encoder(f"{DPR_DIR}/{which}_encoder").to(DEVICE).eval()


@torch.no_grad()
def encode_texts(encoder: Encoder, tokenizer, texts: list[str], batch_size: int, max_len: int) -> np.ndarray:
    encoder.eval()
    device = next(encoder.parameters()).device
    embeddings = []

    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}

        embs = encoder(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            token_type_ids=tokens.get("token_type_ids"),
        )
        embeddings.append(embs.float().cpu().numpy())

        if (start // batch_size) % 100 == 0:
            print(f"  encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)

    if not embeddings:
        return np.empty((0, encoder.model.config.hidden_size), dtype=np.float32)

    return np.concatenate(embeddings, axis=0).astype(np.float32)


def encode_passages(encoder: Encoder, tokenizer, passages: list[dict], batch_size: int, max_len: int) -> np.ndarray:
    texts = [format_passage(p, tokenizer.sep_token) for p in passages]
    return encode_texts(encoder, tokenizer, texts, batch_size, max_len)


def build_index(batch_size: int):
    tokenizer = AutoTokenizer.from_pretrained(DPR_DIR)
    encoder = load_encoder("passage")
    passages = load_passages(PASSAGES_FILE)
    print(f"encoding {len(passages)} passages on {DEVICE}")

    embs = encode_passages(encoder, tokenizer, passages, batch_size, MAX_P_LEN)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"wrote {index.ntotal} vectors to {INDEX_PATH}")


class Retriever:

    def __init__(self, index_path: str = INDEX_PATH, passages_path: str = PASSAGES_FILE):
        self.index = faiss.read_index(index_path)
        self.passages = load_passages(passages_path)
        assert self.index.ntotal == len(self.passages), "index and passages.jsonl are out of sync"

    def retrieve(self, q_emb: np.ndarray, k: int):
        q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)
        scores, ids = self.index.search(q_emb, k)
        doc_embs = self.index.reconstruct_batch(ids.reshape(-1)).reshape(len(q_emb), k, -1)
        passages = [[self.passages[i] for i in row] for row in ids]
        return scores, doc_embs, passages


def evaluate_retrieval(batch_size: int, ks=(1, 5, 20, 100)):
    with open(DEV_FILE, "r", encoding="utf-8") as f:
        dev = [json.loads(line) for line in f if line.strip()]

    tokenizer = AutoTokenizer.from_pretrained(DPR_DIR)
    encoder = load_encoder("question")
    q_emb = encode_texts(encoder, tokenizer, [ex["question"] for ex in dev], batch_size, MAX_Q_LEN)

    retriever = Retriever()
    _, _, retrieved = retriever.retrieve(q_emb, max(ks))

    hit_rank = []
    for ex, passages in zip(dev, retrieved):
        ranks = [r for r, p in enumerate(passages) if has_answer(p["text"], ex["answers"])]
        hit_rank.append(ranks[0] if ranks else float("inf"))

    hit_rank = np.array(hit_rank)
    for k in ks:
        print(f"top-{k} accuracy: {100 * np.mean(hit_rank < k):.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=ENCODE_BATCH_SIZE)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    if args.build:
        build_index(args.batch_size)
    if args.eval:
        evaluate_retrieval(args.batch_size)


if __name__ == "__main__":
    main()
