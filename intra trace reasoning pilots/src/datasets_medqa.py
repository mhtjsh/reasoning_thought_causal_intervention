"""
MedQA dataset loader.
Downloads from HuggingFace or loads from local JSONL.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import MEDQA_DIR, PILOT_CONFIG


@dataclass
class MedQAQuestion:
    """Single MedQA question."""
    question_id: str
    clinical_vignette: str
    options: dict
    correct_answer: str

    @property
    def options_text(self) -> str:
        return "\n".join(f"{k}) {v}" for k, v in sorted(self.options.items()))

    @property
    def correct_answer_text(self) -> str:
        return self.options.get(self.correct_answer, "?")


def load_from_jsonl(filepath: Path) -> List[MedQAQuestion]:
    """Load from cached JSONL."""
    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            q = MedQAQuestion(
                question_id=d.get("question_id", f"medqa_{len(questions):04d}"),
                clinical_vignette=d["question"],
                options=d["options"],
                correct_answer=d["answer_idx"],
            )
            questions.append(q)
    return questions


def load_from_huggingface(split: str = "test") -> List[MedQAQuestion]:
    """Load MedQA 4-option English from HuggingFace."""
    from datasets import load_dataset

    print(f"Downloading MedQA from HuggingFace (split={split})...")
    ds = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", split=split)

    questions = []
    for idx, item in enumerate(ds):
        # Parse options from bigbio format
        opts = {}
        for opt in item.get("options", []):
            opts[opt["key"]] = opt["value"]

        # If options are empty, try alternate format
        if not opts and "answer" in item:
            # Some versions have different formats
            for key in ["A", "B", "C", "D"]:
                if f"option_{key.lower()}" in item:
                    opts[key] = item[f"option_{key.lower()}"]

        answer = item.get("answer_idx", item.get("answer", "A"))

        q = MedQAQuestion(
            question_id=f"medqa_{idx:04d}",
            clinical_vignette=item["question"],
            options=opts,
            correct_answer=answer,
        )
        questions.append(q)

    print(f"Loaded {len(questions)} questions from HuggingFace")
    return questions


def save_to_jsonl(questions: List[MedQAQuestion], filepath: Path):
    """Save to JSONL for caching."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for q in questions:
            d = {
                "question_id": q.question_id,
                "question": q.clinical_vignette,
                "options": q.options,
                "answer_idx": q.correct_answer,
            }
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Cached {len(questions)} questions → {filepath}")


def load_medqa() -> List[MedQAQuestion]:
    """Load MedQA, using cache if available."""
    cache = MEDQA_DIR / "test.jsonl"
    if cache.exists():
        print(f"Loading MedQA from cache: {cache}")
        return load_from_jsonl(cache)
    else:
        qs = load_from_huggingface()
        save_to_jsonl(qs, cache)
        return qs


def select_pilot_sample(
    questions: List[MedQAQuestion],
    n: Optional[int] = None,
    seed: int = 42,
) -> List[MedQAQuestion]:
    """Random pilot sample."""
    n = n or PILOT_CONFIG.n_pilot_questions
    if n >= len(questions):
        return questions
    rng = random.Random(seed)
    sample = rng.sample(questions, n)
    print(f"Selected {len(sample)} pilot questions (seed={seed})")
    return sample


if __name__ == "__main__":
    qs = load_medqa()
    print(f"Total: {len(qs)}")
    pilot = select_pilot_sample(qs)
    save_to_jsonl(pilot, MEDQA_DIR / "pilot_sample.jsonl")
    print(f"\nExample: {pilot[0].clinical_vignette[:150]}...")
    print(f"Options: {pilot[0].options}")
    print(f"Correct: {pilot[0].correct_answer}")
