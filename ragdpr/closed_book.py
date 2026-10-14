import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration

from ragdpr.config import (
    CLOSED_BOOK_DIR, DEVICE, GENERATOR_MODEL, MAX_ANSWER_LEN, MAX_Q_LEN, RAG_BATCH_SIZE, RAG_EPOCHS, RAG_LR,
    TRAIN_FILE,
)
from ragdpr.rag.train import QADataset, collate


def train(args):
    use_amp = args.fp16 and DEVICE == "cuda"
    if args.fp16 and not use_amp:
        print("--fp16 needs a GPU; training in fp32")

    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    model = BartForConditionalGeneration.from_pretrained(GENERATOR_MODEL).to(DEVICE)
    model.train()

    loader = DataLoader(QADataset(TRAIN_FILE), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    total_steps = len(loader) * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=RAG_LR)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    for epoch in range(args.epochs):
        for questions, answers in loader:
            inputs = tokenizer(
                questions, padding=True, truncation=True, max_length=MAX_Q_LEN, return_tensors="pt"
            ).to(DEVICE)
            labels = tokenizer(
                answers, padding=True, truncation=True, max_length=MAX_ANSWER_LEN, return_tensors="pt"
            ).input_ids.to(DEVICE)
            labels[labels == tokenizer.pad_token_id] = -100

            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=use_amp):
                loss = model(**inputs, labels=labels).loss

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

    Path(CLOSED_BOOK_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CLOSED_BOOK_DIR)
    tokenizer.save_pretrained(CLOSED_BOOK_DIR)
    print(f"saved closed-book BART to {CLOSED_BOOK_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=RAG_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=RAG_EPOCHS)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--fp16", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
