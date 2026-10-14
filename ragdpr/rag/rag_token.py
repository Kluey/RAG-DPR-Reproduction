import torch

from ragdpr.config import MAX_ANSWER_LEN
from ragdpr.rag.model import RAG


def rag_token_loss(doc_scores: torch.Tensor, logprobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_logprobs = torch.logsumexp(doc_scores.log_softmax(-1)[:, :, None] + logprobs, dim=1)
    return -(token_logprobs * mask).sum(-1).mean()


@torch.no_grad()
def rag_token_generate(model: RAG, questions: list[str], k: int = 5, max_len: int = MAX_ANSWER_LEN) -> list[str]:
    doc_scores, passages = model.retrieve(questions, k)
    doc_logprobs = doc_scores.log_softmax(-1)
    inputs = model.tokenize_inputs(model.build_generator_inputs(questions, passages))
    encoder_outputs = model.generator.get_encoder()(
        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
    )

    config = model.generator.config
    B = len(questions)
    prefix = torch.full((B, 1), config.decoder_start_token_id, device=model.device)
    finished = torch.zeros(B, dtype=torch.bool, device=model.device)

    for _ in range(max_len):
        logits = model.generator(
            encoder_outputs=encoder_outputs,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=prefix.repeat_interleave(k, dim=0),
        ).logits[:, -1]
        next_logprobs = logits.float().log_softmax(-1).view(B, k, -1)
        marginal = torch.logsumexp(doc_logprobs[:, :, None] + next_logprobs, dim=1)

        next_token = marginal.argmax(-1)
        next_token = torch.where(finished, torch.full_like(next_token, config.pad_token_id), next_token)
        prefix = torch.cat([prefix, next_token[:, None]], dim=1)
        finished |= next_token == config.eos_token_id
        if finished.all():
            break

    return [s.strip() for s in model.g_tokenizer.batch_decode(prefix, skip_special_tokens=True)]
