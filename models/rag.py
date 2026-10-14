from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right

from config import BERT_MODEL, GENERATOR_MODEL, MAX_ANSWER_LEN, MAX_GEN_INPUT_LEN, MAX_Q_LEN, N_DOCS
from retrieval.dpr.index import Retriever
from retrieval.dpr.model import Encoder


class RAG(nn.Module):
    def __init__(
        self,
        question_encoder_path: str,
        retriever: Retriever,
        generator_name: str = GENERATOR_MODEL,
        n_docs: int = N_DOCS,
    ):
        super().__init__()
        self.question_encoder = Encoder(question_encoder_path)
        self.q_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        self.generator = BartForConditionalGeneration.from_pretrained(generator_name)
        self.g_tokenizer = AutoTokenizer.from_pretrained(generator_name)
        self.retriever = retriever
        self.n_docs = n_docs

    @classmethod
    def load(cls, checkpoint_dir: str, retriever: Retriever, n_docs: int = N_DOCS) -> "RAG":
        return cls(f"{checkpoint_dir}/question_encoder", retriever, f"{checkpoint_dir}/generator", n_docs)

    def save(self, checkpoint_dir: str):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.question_encoder.model.save_pretrained(f"{checkpoint_dir}/question_encoder")
        self.generator.save_pretrained(f"{checkpoint_dir}/generator")
        self.g_tokenizer.save_pretrained(f"{checkpoint_dir}/generator")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def retrieve(self, questions: list[str], k: int | None = None):
        k = k or self.n_docs
        tokens = self.q_tokenizer(
            questions, padding=True, truncation=True, max_length=MAX_Q_LEN, return_tensors="pt"
        ).to(self.device)
        q = self.question_encoder(**tokens)
        _, doc_embs, passages = self.retriever.retrieve(q.detach().float().cpu().numpy(), k)
        doc_embs = torch.from_numpy(doc_embs).to(self.device)
        doc_scores = (q.float()[:, None, :] * doc_embs).sum(-1)
        return doc_scores, passages

    @staticmethod
    def build_generator_inputs(questions: list[str], passages: list[list[dict]]) -> list[str]:
        return [f"{p['title']} / {p['text']} // {q}" for q, docs in zip(questions, passages) for p in docs]

    def tokenize_inputs(self, gen_inputs: list[str]):
        return self.g_tokenizer(
            gen_inputs, padding=True, truncation=True, max_length=MAX_GEN_INPUT_LEN, return_tensors="pt"
        ).to(self.device)

    def token_logprobs(self, gen_inputs: list[str], answers: list[str], k: int):
        inputs = self.tokenize_inputs(gen_inputs)
        targets = self.g_tokenizer(
            answers, padding=True, truncation=True, max_length=MAX_ANSWER_LEN, return_tensors="pt"
        ).to(self.device)
        labels = targets.input_ids.repeat_interleave(k, dim=0)
        config = self.generator.config
        decoder_input_ids = shift_tokens_right(labels, config.pad_token_id, config.decoder_start_token_id)

        logits = self.generator(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=decoder_input_ids,
        ).logits
        logprobs = logits.float().log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
        return logprobs.view(len(answers), k, -1), targets.attention_mask
