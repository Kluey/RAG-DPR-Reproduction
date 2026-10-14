import re
import string


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold_answers: list[str]) -> bool:
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(a) for a in gold_answers)


def has_answer(passage_text: str, gold_answers: list[str]) -> bool:
    passage = f" {normalize_answer(passage_text)} "
    for a in gold_answers:
        answer = normalize_answer(a)
        if answer and f" {answer} " in passage:
            return True
    return False
