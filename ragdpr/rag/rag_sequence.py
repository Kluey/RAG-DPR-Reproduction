import torch

from ragdpr.config import MAX_ANSWER_LEN, NUM_BEAMS
from ragdpr.rag.model import RAG


def rag_sequence_loss(doc_scores: torch.Tensor, logprobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    seq_logprobs = (logprobs * mask[:, None, :]).sum(-1)
    return -torch.logsumexp(doc_scores.log_softmax(-1) + seq_logprobs, dim=-1).mean()


@torch.no_grad()
def rag_sequence_generate(
    model: RAG, questions: list[str], k: int = 5, num_beams: int = NUM_BEAMS, max_len: int = MAX_ANSWER_LEN
) -> list[str]:
    doc_scores, passages = model.retrieve(questions, k)
    doc_logprobs = doc_scores.log_softmax(-1)
    gen_inputs = model.build_generator_inputs(questions, passages)

    inputs = model.tokenize_inputs(gen_inputs)
    out = model.generator.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        max_length=max_len,
    )
    decoded = model.g_tokenizer.batch_decode(out, skip_special_tokens=True)
    per_question = k * num_beams

    predictions = []
    for b in range(len(questions)):
        candidates = list(dict.fromkeys(c.strip() for c in decoded[b * per_question : (b + 1) * per_question]))
        candidates = [c for c in candidates if c] or [""]

        docs = gen_inputs[b * k : (b + 1) * k]
        logprobs, mask = model.token_logprobs(docs * len(candidates), candidates, k)
        seq_logprobs = (logprobs * mask[:, None, :]).sum(-1)

        scores = torch.logsumexp(doc_logprobs[b][None, :] + seq_logprobs, dim=-1)
        predictions.append(candidates[scores.argmax().item()])
    return predictions
