import argparse
import json

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ragdpr.config import (
    BERT_MODEL, DEVICE, DPR_BATCH_SIZE, DPR_DIR, DPR_EPOCHS, DPR_LR, DPR_MAX_GRAD_NORM, DPR_WARMUP_STEPS,
    MAX_P_LEN, MAX_Q_LEN, TRAIN_FILE,
)
from ragdpr.dpr.model import BiEncoder


class DPRDataset(Dataset):
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            examples = [json.loads(line) for line in f if line.strip()]
        self.examples = [ex for ex in examples if ex["hard_negatives"]]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex["question"], ex["positive"], ex["hard_negatives"][0]


def format_passage(passage: dict, sep_token: str) -> str:
    return f"{passage['title']} {sep_token} {passage['text']}"


class DPRCollator:
    def __init__(self, tokenizer, max_q_len: int = MAX_Q_LEN, max_p_len: int = MAX_P_LEN):
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len

    def __call__(self, batch: list) -> dict:
        questions, positives, negatives = zip(*batch)
        sep = self.tokenizer.sep_token
        passages = [format_passage(p, sep) for p in positives + negatives]
        tokenize = lambda texts, max_len: self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        return {
            "q_inputs": tokenize(questions, self.max_q_len),
            "p_inputs": tokenize(passages, self.max_p_len),
            "positive_idx": torch.arange(len(batch)),
        }


def train(args):
    use_amp = args.fp16 and DEVICE == "cuda"
    if args.fp16 and not use_amp:
        print("--fp16 needs a GPU; training in fp32")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = BiEncoder().to(DEVICE)
    model.train()

    loader = DataLoader(
        DPRDataset(TRAIN_FILE), batch_size=args.batch_size, shuffle=True, collate_fn=DPRCollator(tokenizer)
    )
    total_steps = len(loader) * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=DPR_LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, DPR_WARMUP_STEPS, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            q_inputs = {k: v.to(DEVICE) for k, v in batch["q_inputs"].items()}
            p_inputs = {k: v.to(DEVICE) for k, v in batch["p_inputs"].items()}
            positive_idx = batch["positive_idx"].to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=use_amp):
                scores = model(q_inputs, p_inputs)
            loss = BiEncoder.loss(scores.float(), positive_idx)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), DPR_MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            if step % 50 == 0 or step == 1:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f}", flush=True)
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    model.question_encoder.model.save_pretrained(f"{DPR_DIR}/question_encoder")
    model.passage_encoder.model.save_pretrained(f"{DPR_DIR}/passage_encoder")
    tokenizer.save_pretrained(DPR_DIR)
    print(f"saved encoders + tokenizer to {DPR_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=DPR_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DPR_EPOCHS)
    parser.add_argument("--max_steps", type=int, default=None, help="stop early (smoke tests)")
    parser.add_argument("--fp16", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
