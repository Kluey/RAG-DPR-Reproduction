import argparse
import json

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

from ragdpr.config import (
    CLOSED_BOOK_DIR, DEV_FILE, DEVICE, EVAL_BATCH_SIZE, INDEX_PATH, MAX_ANSWER_LEN, MAX_Q_LEN, N_DOCS,
    NUM_BEAMS, PASSAGES_FILE, RAG_DIR,
)
from ragdpr.metrics import exact_match
from ragdpr.results import save_result
from ragdpr.rag.model import RAG
from ragdpr.rag.rag_sequence import rag_sequence_generate
from ragdpr.rag.rag_token import rag_token_generate
from ragdpr.dpr.index import Retriever

GENERATORS = {"seq": rag_sequence_generate, "token": rag_token_generate}


def load_dev(path: str, limit: int | None = None) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        dev = [json.loads(line) for line in f if line.strip()]
    return dev[:limit] if limit is not None else dev


@torch.no_grad()
def generate_no_retrieval(generator_dir: str, questions: list[str]) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(generator_dir)
    generator = BartForConditionalGeneration.from_pretrained(generator_dir).to(DEVICE).eval()
    predictions = []
    for start in range(0, len(questions), EVAL_BATCH_SIZE):
        inputs = tokenizer(
            questions[start : start + EVAL_BATCH_SIZE],
            padding=True, truncation=True, max_length=MAX_Q_LEN, return_tensors="pt",
        ).to(DEVICE)
        out = generator.generate(**inputs, num_beams=NUM_BEAMS, max_length=MAX_ANSWER_LEN)
        predictions += [s.strip() for s in tokenizer.batch_decode(out, skip_special_tokens=True)]
    return predictions


def evaluate(args):
    dev = load_dev(DEV_FILE, args.limit)
    questions = [ex["question"] for ex in dev]

    if args.no_retrieval:
        predictions = generate_no_retrieval(CLOSED_BOOK_DIR, questions)
    else:
        model = RAG.load(RAG_DIR.format(variant=args.variant), Retriever(INDEX_PATH, PASSAGES_FILE), args.n_docs)
        model.to(DEVICE).eval()
        generate = GENERATORS[args.variant]
        predictions = []
        for start in range(0, len(questions), EVAL_BATCH_SIZE):
            predictions += generate(model, questions[start : start + EVAL_BATCH_SIZE], k=args.n_docs)
            print(f"  {len(predictions)}/{len(questions)}", flush=True)

    em = [exact_match(pred, ex["answers"]) for pred, ex in zip(predictions, dev)]
    name = "closed-book BART" if args.no_retrieval else f"RAG-{args.variant}"
    score = round(100 * sum(em) / len(em), 1)
    print(f"{name}: EM = {score:.1f}% ({sum(em)}/{len(em)})")
    key = "closed_book" if args.no_retrieval else f"rag_{args.variant}"
    save_result(["exact_match", key], {"n": len(em), "em": score})

    if args.predictions_out:
        records = [
            {"question": ex["question"], "answers": ex["answers"], "prediction": pred, "exact_match": hit}
            for ex, pred, hit in zip(dev, predictions, em)
        ]
        with open(args.predictions_out, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(GENERATORS), default="seq")
    parser.add_argument("--n_docs", type=int, default=N_DOCS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_retrieval", action="store_true")
    parser.add_argument("--predictions_out", default=None)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
