import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from config import BERT_MODEL

class Encoder(nn.Module):
    def __init__(self, model_name: str = BERT_MODEL):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return out.last_hidden_state[:, 0] 


class BiEncoder(nn.Module):
    def __init__(self, shared: bool = False):
        super().__init__()
        self.question_encoder = Encoder()
        self.passage_encoder = self.question_encoder if shared else Encoder()

    def encode_questions(self, **inputs):
        return self.question_encoder(**inputs)

    def encode_passages(self, **inputs):
        return self.passage_encoder(**inputs)

    def forward(self, q_inputs: dict, p_inputs: dict):
        q = self.encode_questions(**q_inputs)  
        p = self.encode_passages(**p_inputs)   
        return q @ p.T                         

    @staticmethod
    def loss(scores: torch.Tensor, positive_idx: torch.Tensor):
        return F.cross_entropy(scores, positive_idx)
