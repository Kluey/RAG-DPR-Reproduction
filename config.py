import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_NAME = "Tevatron/wikipedia-nq"
DATA_DIR = "data/nq"
TRAIN_FILE = f"{DATA_DIR}/train.jsonl"
DEV_FILE = f"{DATA_DIR}/dev.jsonl"
PASSAGES_FILE = f"{DATA_DIR}/passages.jsonl"
N_TRAIN = 10000
N_DEV = 1000
N_HARD_NEGATIVES = 1

BERT_MODEL = "bert-base-uncased"
GENERATOR_MODEL = "facebook/bart-base"
MAX_Q_LEN = 64
MAX_P_LEN = 256
MAX_GEN_INPUT_LEN = 512
MAX_ANSWER_LEN = 32

DPR_DIR = "checkpoints/dpr"
INDEX_PATH = "checkpoints/index/index.faiss"
RAG_DIR = "checkpoints/rag_{variant}"
CLOSED_BOOK_DIR = "checkpoints/closed_book"

DPR_BATCH_SIZE = 16
DPR_EPOCHS = 10
DPR_LR = 2e-5
DPR_WARMUP_STEPS = 100
DPR_MAX_GRAD_NORM = 2.0

ENCODE_BATCH_SIZE = 128

N_DOCS = 5
RAG_BATCH_SIZE = 4
RAG_EPOCHS = 3
RAG_LR = 3e-5
NUM_BEAMS = 4
EVAL_BATCH_SIZE = 8
