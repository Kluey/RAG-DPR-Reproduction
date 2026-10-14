import argparse
import json

import torch
from torch.utils.data import DataLoader, Dataset

from ragdpr.config import (
    DEVICE, DPR_DIR, GENERATOR_MODEL, INDEX_PATH, N_DOCS, PASSAGES_FILE, RAG_BATCH_SIZE, RAG_DIR, RAG_EPOCHS, RAG_LR,
    TRAIN_FILE,
)
from ragdpr.rag.model import RAG
from ragdpr.rag.rag_sequence import rag_sequence_loss
from ragdpr.rag.rag_token import rag_token_loss
from ragdpr.dpr.index import Retriever

LOSSES = {"seq": rag_sequence_loss, "token": rag_token_loss}


class QADataset(Dataset):
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            examples = [json.loads(line) for line in f if line.strip()]
        self.examples = [(ex["question"], ex["answers"][0]) for ex in examples if ex["answers"]]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    questions, answers = zip(*batch)
    return list(questions), list(answers)


def train(args):
    use_amp = args.fp16 and DEVICE == "cuda"
    if args.fp16 and not use_amp:
        print("--fp16 needs a GPU; training in fp32")

    retriever = Retriever(INDEX_PATH, PASSAGES_FILE)
    model = RAG(f"{DPR_DIR}/question_encoder", retriever, GENERATOR_MODEL, args.n_docs).to(DEVICE)
    model.train()

    loader = DataLoader(QADataset(TRAIN_FILE), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    total_steps = len(loader) * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=RAG_LR)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = LOSSES[args.variant]

    step = 0
    for epoch in range(args.epochs):
        for questions, answers in loader:
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=use_amp):
                doc_scores, passages = model.retrieve(questions, args.n_docs)
                gen_inputs = model.build_generator_inputs(questions, passages)
                logprobs, mask = model.token_logprobs(gen_inputs, answers, args.n_docs)
            loss = loss_fn(doc_scores.float(), logprobs.float(), mask.float())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            if step % 50 == 0 or step == 1:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f}", flush=True)
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    output_dir = RAG_DIR.format(variant=args.variant)
    model.save(output_dir)
    print(f"saved question encoder + generator to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(LOSSES), default="seq")
    parser.add_argument("--n_docs", type=int, default=N_DOCS)
    parser.add_argument("--batch_size", type=int, default=RAG_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=RAG_EPOCHS)
    parser.add_argument("--max_steps", type=int, default=None, help="stop early (smoke tests)")
    parser.add_argument("--fp16", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
