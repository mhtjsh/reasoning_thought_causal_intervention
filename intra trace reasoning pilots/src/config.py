# -*- coding: utf-8 -*-
"""
Configuration for Intra-Trace Reasoning Uncertainty Pilots.
All model IDs, GPU mappings, thresholds, and paths.

Redesigned for:
- Plain HuggingFace transformers (no vLLM dependency)
- Robust <think> extraction that handles models without think tags
- Repetition detection and truncation
- Separate pilot scripts
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/home/mjoshi/rlm and ttt for uncertain medical reasoning")
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MEDQA_DIR = DATA_DIR / "medqa"
TRACES_DIR = DATA_DIR / "traces"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = DATA_DIR / "pilot_results"

# Create directories
for d in [DATA_DIR, MEDQA_DIR, TRACES_DIR, EMBEDDINGS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Reasoning Models ────────────────────────────────────────────────────────

@dataclass
class ReasoningModelConfig:
    """Configuration for a reasoning model."""
    name: str
    hf_id: str
    # Whether model natively uses <think> tags in output
    has_think_tags: bool
    # Whether to enable thinking mode via chat template
    enable_thinking: bool = True
    # Device mapping strategy
    device_map: str = "auto"
    # Generation parameters
    max_new_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    # Repetition penalty to prevent degenerate loops
    repetition_penalty: float = 1.1


# Models verified to work on 3x A6000 (144GB total)
# NOTE: Qwen3.5-27B requires bleeding-edge transformers from git main —
#       use generate_traces_qwen35.py separately with its own deps.
REASONING_MODELS: Dict[str, ReasoningModelConfig] = {
    "qwq-32b": ReasoningModelConfig(
        name="qwq-32b",
        hf_id="Qwen/QwQ-32B",
        has_think_tags=False,  # QwQ outputs thinking directly, NO <think> tags
        enable_thinking=False,
        max_new_tokens=8192,
        repetition_penalty=1.15,  # Higher to combat the severe repetition
    ),
    "qwen3-8b": ReasoningModelConfig(
        name="qwen3-8b",
        hf_id="Qwen/Qwen3-8B",
        has_think_tags=True,  # Qwen3 uses <think>...</think>
        enable_thinking=True,
        max_new_tokens=8192,
        repetition_penalty=1.1,
    ),
    "qwen3-4b-thinking": ReasoningModelConfig(
        name="qwen3-4b-thinking",
        hf_id="Qwen/Qwen3-4B-Thinking-2507",
        has_think_tags=True,
        enable_thinking=True,
        max_new_tokens=8192,
        repetition_penalty=1.1,
    ),
    # Qwen3.5-27B: requires transformers from git main (Gated DeltaNet arch)
    # Run via generate_traces_qwen35.py with requirements_qwen35.txt
    "qwen3.5-27b": ReasoningModelConfig(
        name="qwen3.5-27b",
        hf_id="Qwen/Qwen3.5-27B",
        has_think_tags=True,  # Uses <think>...</think> by default
        enable_thinking=True,
        max_new_tokens=8192,  # Model card recommends 8192 for most queries
        temperature=1.0,       # Model card recommended sampling
        top_p=0.95,
        repetition_penalty=1.0,  # Uses presence_penalty instead (handled in script)
    ),
}


# ─── Embedding Models ────────────────────────────────────────────────────────

@dataclass
class EmbeddingModelConfig:
    """Configuration for an embedding model."""
    name: str
    hf_id: str
    max_seq_length: int = 512
    approx_vram_gb: float = 1.0


EMBEDDING_MODELS: Dict[str, EmbeddingModelConfig] = {
    "bge-m3": EmbeddingModelConfig(
        name="bge-m3",
        hf_id="BAAI/bge-m3",
        max_seq_length=512,
        approx_vram_gb=1.2,
    ),
    "qwen3-embedding-8b": EmbeddingModelConfig(
        name="qwen3-embedding-8b",
        hf_id="Qwen/Qwen3-Embedding-8B",
        max_seq_length=512,
        approx_vram_gb=16.0,
    ),
}


# ─── Pilot Thresholds ────────────────────────────────────────────────────────

@dataclass
class PilotConfig:
    """Thresholds and limits for pilot experiments."""
    n_pilot_questions: int = 20
    # Pilot 1: Hypothesis transition detection
    cosine_drop_threshold: float = 0.15
    min_sentence_words: int = 5
    pilot1_precision_threshold: float = 0.6
    pilot1_recall_threshold: float = 0.5
    # Pilot 2: Dynamics predict correctness
    pilot2_p_value_threshold: float = 0.05
    # Pilot 3: Uncertainty marker calibration
    pilot3_spearman_threshold: float = 0.15
    pilot3_min_models_passing: int = 2


PILOT_CONFIG = PilotConfig()


# ─── Prompts ──────────────────────────────────────────────────────────────────

MEDQA_PROMPT_TEMPLATE = """You are a medical expert. Based on the following clinical case, what is the most likely diagnosis?

{clinical_vignette}

{options_text}"""

MATH_PROMPT_TEMPLATE = """Solve the following problem.

{problem_statement}"""


# ─── Repetition Detection ────────────────────────────────────────────────────

def remove_repetitions(text: str, min_repeat_len: int = 50) -> str:
    """
    Detect and remove degenerate repetitions in model output.

    QwQ-32B has a known issue where it repeats its final answer
    section 100+ times until hitting the token limit. This function
    detects such patterns and truncates to the first occurrence.

    Args:
        text: Raw model output.
        min_repeat_len: Minimum length of repeated block to detect.

    Returns:
        Cleaned text with repetitions removed.
    """
    if len(text) < min_repeat_len * 2:
        return text

    # Strategy 1: Detect repeated "### Final Answer" blocks
    final_answer_pattern = r"(###?\s*Final\s*Answer.*?)(?=###?\s*Final\s*Answer|\Z)"
    final_matches = list(re.finditer(final_answer_pattern, text, re.DOTALL | re.IGNORECASE))
    if len(final_matches) > 2:
        # Keep everything up to and including the first "Final Answer" block
        first_end = final_matches[0].end()
        return text[:first_end].strip()

    # Strategy 2: Detect repeated \\boxed{X} blocks
    boxed_pattern = r"(\\\\boxed\{[^}]+\})"
    boxed_matches = list(re.finditer(boxed_pattern, text))
    if len(boxed_matches) > 3:
        # Keep up to end of second boxed (some legitimate repeat once)
        return text[:boxed_matches[1].end()].strip()

    # Strategy 3: Detect repeated large blocks (>50 chars repeated 3+ times)
    # Find the longest repeated substring
    for block_len in range(200, min_repeat_len - 1, -10):
        for start in range(0, len(text) - block_len * 2):
            block = text[start:start + block_len]
            count = text.count(block)
            if count >= 3:
                # Found a repeated block — keep up to first 2 occurrences
                second_occurrence = text.find(block, start + block_len)
                if second_occurrence > 0:
                    return text[:second_occurrence + block_len].strip()

    return text


# ─── Thinking Extraction ─────────────────────────────────────────────────────

def extract_thinking_and_answer(raw_output: str, has_think_tags: bool) -> tuple:
    """
    Extract thinking trace and final answer from model output.

    Handles two output formats:
    1. Models with <think>...</think> tags (Qwen3, Qwen3-4B-Thinking)
    2. Models without tags (QwQ-32B) — outputs thinking then answer directly

    Args:
        raw_output: Full model output text.
        has_think_tags: Whether the model uses <think> tags.

    Returns:
        (thinking_text, final_answer_text)
    """
    # First, remove degenerate repetitions
    raw_output = remove_repetitions(raw_output)

    if has_think_tags:
        # ── Models with <think>...</think> ──
        think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            answer = raw_output[think_match.end():].strip()
            return thinking, answer

    # ── Models without <think> tags (e.g., QwQ-32B) ──
    # QwQ outputs: thinking text → then final answer starting with
    # a pattern like "The answer is..." or "**Answer: ..."
    # Strategy: Find where the "answer declaration" starts

    # Pattern: Look for the FIRST definitive answer statement
    answer_patterns = [
        r"\nThe most likely (?:diagnosis|answer|cause)",
        r"\n\*\*(?:Answer|Final Answer|The answer)",
        r"\n#{1,3}\s*(?:Answer|Final Answer|Conclusion|Key Points)",
    ]

    best_split = len(raw_output)
    for pattern in answer_patterns:
        match = re.search(pattern, raw_output)
        if match and match.start() < best_split and match.start() > 100:
            best_split = match.start()

    if best_split < len(raw_output):
        thinking = raw_output[:best_split].strip()
        answer = raw_output[best_split:].strip()
    else:
        # Fallback: split at last double newline
        parts = raw_output.rsplit("\n\n", 1)
        thinking = parts[0].strip() if len(parts) > 1 else raw_output
        answer = parts[-1].strip()

    return thinking, answer


def extract_predicted_option(text: str) -> Optional[str]:
    """
    Extract the predicted MCQ option (A/B/C/D/E) from answer text.

    Robustly handles:
    - Boxed answers: \\boxed{A}
    - Explicit statements: "The answer is A"
    - Bold answers: **A)**
    - Standalone letters at start/end
    """
    # Clean repetitions first
    text = remove_repetitions(text)

    # Pattern 1: \\boxed{X}
    match = re.search(r"\\boxed\{([A-E])\}", text)
    if match:
        return match.group(1).upper()

    # Pattern 2: "The answer is X" / "Answer: X"
    match = re.search(
        r"(?:the answer is|answer:\s*|answer is)\s*\*?\*?\(?([A-E])\)?",
        text, re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # Pattern 3: Bold answer "**A)**" or "**A**"
    match = re.search(r"\*\*\(?([A-E])\)?\*\*", text)
    if match:
        return match.group(1).upper()

    # Pattern 4: First line starts with an option
    first_line = text.strip().split("\n")[0].strip()
    match = re.match(r"^\*?\*?\(?([A-E])\)?[\.\)\s:\*]", first_line)
    if match:
        return match.group(1).upper()

    # Pattern 5: Last standalone letter mention in first 500 chars
    short_text = text[:500]
    matches = re.findall(r"\b([A-E])\b", short_text)
    if matches:
        return matches[-1].upper()

    return None


# ─── Uncertainty Markers (for Pilot 3) ────────────────────────────────────────

UNCERTAINTY_MARKERS = {
    "hedge": [
        r"\bI think\b",
        r"\bprobably\b",
        r"\bmight be\b",
        r"\bperhaps\b",
        r"\blikely\b",
        r"\bI'm not sure\b",
        r"\bit seems\b",
        r"\bpossibly\b",
        r"\bcould be\b",
    ],
    "revise": [
        r"\bWait\b",
        r"\bActually\b",
        r"\bNo,\b",
        r"\bLet me redo\b",
        r"\bthat's wrong\b",
        r"\bcorrection\b",
        r"\bLet me reconsider\b",
        r"\bOn second thought\b",
    ],
    "verify": [
        r"\bLet me verify\b",
        r"\bdouble-check\b",
        r"\bchecking\b",
        r"\bto confirm\b",
        r"\bLet me check\b",
    ],
    "explicit": [
        r"\bI'm uncertain\b",
        r"\bnot confident\b",
        r"\bunsure\b",
        r"\bhard to determine\b",
        r"\bdifficult to say\b",
        r"\bnot entirely sure\b",
    ],
}
