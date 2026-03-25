# -*- coding: utf-8 -*-
"""
Generate reasoning traces using Qwen3.5-27B.

Qwen3.5-27B uses:
- Gated DeltaNet + sparse MoE architecture
- <think>...</think> tags for reasoning by default
- Recommended: temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5

Usage:
    python generate_traces_qwen35.py              # 20 questions
    python generate_traces_qwen35.py --n 5        # 5 questions
"""

import argparse
import json
import time
import re
import sys
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TRACES_DIR,
    MEDQA_DIR,
    MEDQA_PROMPT_TEMPLATE,
    extract_thinking_and_answer,
    extract_predicted_option,
    remove_repetitions,
)
from datasets_medqa import load_medqa, select_pilot_sample, load_from_jsonl, save_to_jsonl


# ---- Qwen3.5-27B specific config ----

MODEL_ID = "Qwen/Qwen3.5-27B"
MODEL_NAME = "qwen3.5-27b"

# From model card: recommended sampling for thinking mode on general tasks
GENERATION_KWARGS = dict(
    max_new_tokens=8192,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    do_sample=True,
    # presence_penalty=1.5 ≈ repetition_penalty ~1.3 for this model
    repetition_penalty=1.3,
)


@dataclass
class ReasoningTrace:
    """A single reasoning trace (same schema as generate_traces.py)."""
    question_id: str
    model_name: str
    raw_output: str
    thinking_text: str
    final_answer: str
    predicted_option: Optional[str]
    correct_option: str
    is_correct: bool
    generation_time_s: float
    output_tokens: int
    thinking_char_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def check_deltannet_deps():
    """
    Check that the critical Gated DeltaNet dependencies are installed.
    Without these, Qwen3.5 falls back to pure-torch CPU compute (~100x slower).
    Non-interactive — logs warning and continues.
    """
    missing = []
    try:
        import causal_conv1d
        print(f"  causal-conv1d: {causal_conv1d.__version__}")
    except ImportError:
        missing.append("causal-conv1d")

    try:
        import fla
        print(f"  flash-linear-attention: installed")
    except ImportError:
        missing.append("flash-linear-attention")

    if missing:
        print(f"\n{'!'*60}")
        print(f"WARNING: Missing libraries: {', '.join(missing)}")
        print(f"Without these, Qwen3.5 Gated DeltaNet layers fall back to")
        print(f"pure-torch implementation = ~100x slower (CPU-bound).")
        print(f"")
        print(f"Install with:")
        print(f"  pip install causal-conv1d>=1.4.0 flash-linear-attention --no-build-isolation")
        print(f"{'!'*60}")
        print("Continuing in fallback mode...\n")


def load_model():
    """
    Load Qwen3.5-27B with proper handling for the new architecture.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Loading: {MODEL_ID}")
    print(f"  torch.cuda.device_count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")
    print(f"{'='*60}")

    # Check DeltaNet dependencies BEFORE loading model
    check_deltannet_deps()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        padding_side="left",
    )

    # Try loading with flash attention first
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            attn_implementation="flash_attention_2",
            **load_kwargs,
        )
        print("  Loaded with flash_attention_2")
    except Exception as e:
        print(f"  Flash attention failed ({e}), loading without it...")
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
            print("  Loaded with default attention")
        except torch.cuda.OutOfMemoryError:
            print("  OOM! Trying 4-bit quantization...")
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
            print("  Loaded with 4-bit quantization")

    model.eval()
    print(f"  Model loaded successfully")
    return model, tokenizer


def build_prompt(question, tokenizer) -> str:
    """Build prompt with Qwen3.5 chat template."""
    prompt_text = MEDQA_PROMPT_TEMPLATE.format(
        clinical_vignette=question.clinical_vignette,
        options_text=question.options_text,
    )

    messages = [{"role": "user", "content": prompt_text}]

    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return formatted
    except TypeError:
        # Older chat template without enable_thinking
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return formatted
        except Exception as e:
            print(f"    Chat template failed ({e}), using raw prompt")
            return prompt_text


def _extract_think_from_raw(raw_output: str) -> tuple:
    """
    Extract <think> content handling both cases:
    1. Tags preserved in output (skip_special_tokens=False or tags not special)
    2. Tags stripped by tokenizer (fall back to heuristic)
    """
    # Try explicit <think>...</think> tags first
    think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = raw_output[think_match.end():].strip()
        # Clean any remaining special tokens from answer
        answer = re.sub(r"<\|[^>]+\|>", "", answer).strip()
        return thinking, answer

    # Fall back to the general extract function
    return extract_thinking_and_answer(raw_output, has_think_tags=True)


def generate_single(model, tokenizer, question) -> ReasoningTrace:
    """Generate a trace for a single question."""
    prompt = build_prompt(question, tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    start = time.time()
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **GENERATION_KWARGS)
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"    OOM on {question.question_id}, retrying with 4096 tokens...")
        reduced_kwargs = {**GENERATION_KWARGS, "max_new_tokens": 4096}
        with torch.no_grad():
            outputs = model.generate(**inputs, **reduced_kwargs)
    gen_time = time.time() - start

    # Decode new tokens — try to preserve <think> tags
    new_tokens = outputs[0][input_len:]
    output_tokens = len(new_tokens)

    # First try with skip_special_tokens=False to keep <think> tags
    raw_output_full = tokenizer.decode(new_tokens, skip_special_tokens=False)
    # Clean chat-template special tokens but keep <think>/<\think>
    raw_output_clean = re.sub(r"<\|[^>]+\|>", "", raw_output_full).strip()

    # If <think> tags are present, use that version
    if "<think>" in raw_output_clean:
        raw_output = raw_output_clean
    else:
        # Tags were treated as special tokens, use clean decode
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Extract thinking and answer
    thinking, answer = _extract_think_from_raw(raw_output)
    thinking = remove_repetitions(thinking)
    predicted = extract_predicted_option(answer)

    # Fallback: try extracting from thinking text
    if predicted is None:
        predicted = extract_predicted_option(thinking)

    trace = ReasoningTrace(
        question_id=question.question_id,
        model_name=MODEL_NAME,
        raw_output=remove_repetitions(raw_output),
        thinking_text=thinking,
        final_answer=answer[:3000],
        predicted_option=predicted,
        correct_option=question.correct_answer,
        is_correct=(predicted == question.correct_answer) if predicted else False,
        generation_time_s=gen_time,
        output_tokens=output_tokens,
        thinking_char_count=len(thinking),
    )

    status = "correct" if trace.is_correct else "WRONG"
    parsed = trace.predicted_option or "?"
    print(f"    [{status}] {question.question_id}: "
          f"predicted={parsed} correct={question.correct_answer} "
          f"({gen_time:.1f}s, {output_tokens} tok, "
          f"{len(thinking)} chars thinking)")

    return trace


def main():
    parser = argparse.ArgumentParser(
        description="Generate traces with Qwen3.5-27B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Requirements:
    pip install -r requirements.txt
        """,
    )
    parser.add_argument("--n", type=int, default=20, help="Number of questions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=8192,
                       help=f"Max new tokens to generate (default: 8192)")
    args = parser.parse_args()

    # Apply --max-tokens override
    GENERATION_KWARGS["max_new_tokens"] = args.max_tokens

    # --- Check transformers version ---
    import transformers
    print(f"transformers version: {transformers.__version__}")

    # --- Load questions ---
    pilot_file = MEDQA_DIR / "pilot_sample.jsonl"
    if pilot_file.exists():
        questions = load_from_jsonl(pilot_file)
        print(f"Loaded {len(questions)} pilot questions from cache")
    else:
        all_qs = load_medqa()
        questions = select_pilot_sample(all_qs, n=args.n, seed=args.seed)
        save_to_jsonl(questions, pilot_file)

    if args.n < len(questions):
        questions = questions[:args.n]

    print(f"\nModel: {MODEL_ID}")
    print(f"Questions: {len(questions)}")
    print(f"Max new tokens: {GENERATION_KWARGS['max_new_tokens']}")

    # --- Prepare output ---
    output_dir = TRACES_DIR / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_file = output_dir / "traces.jsonl"

    # Check for existing traces (resume support)
    existing = {}
    if traces_file.exists():
        with open(traces_file, "r") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line.strip())
                    existing[d["question_id"]] = d
        print(f"Found {len(existing)} existing traces (will skip)")

    # --- Load model ---
    model, tokenizer = load_model()

    # --- Generate ---
    traces = []
    for i, q in enumerate(questions):
        if q.question_id in existing:
            print(f"  [{i+1}/{len(questions)}] {q.question_id}: SKIPPED (cached)")
            continue

        print(f"  [{i+1}/{len(questions)}] Generating...")
        try:
            trace = generate_single(model, tokenizer, q)
            traces.append(trace)

            # Save incrementally
            with open(traces_file, "a") as f:
                f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"    ERROR on {q.question_id}: {e}")
            import traceback
            traceback.print_exc()

            trace = ReasoningTrace(
                question_id=q.question_id,
                model_name=MODEL_NAME,
                raw_output=f"ERROR: {str(e)}",
                thinking_text="",
                final_answer="",
                predicted_option=None,
                correct_option=q.correct_answer,
                is_correct=False,
                generation_time_s=0,
                output_tokens=0,
                thinking_char_count=0,
            )
            traces.append(trace)
            with open(traces_file, "a") as f:
                f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

    # --- Summary ---
    all_traces = list(existing.values()) + [t.to_dict() for t in traces]
    n_total = len(all_traces)
    n_correct = sum(1 for t in all_traces
                    if (t.get("is_correct") if isinstance(t, dict) else t.is_correct))
    n_errors = sum(1 for t in all_traces
                   if (t.get("raw_output", "").startswith("ERROR")
                       if isinstance(t, dict)
                       else t.raw_output.startswith("ERROR")))

    print(f"\n{'='*60}")
    print(f"RESULTS: {MODEL_NAME}")
    print(f"  Total:    {n_total}")
    print(f"  Correct:  {n_correct}/{n_total} ({100*n_correct/max(n_total,1):.1f}%)")
    print(f"  Errors:   {n_errors}")
    print(f"  Saved to: {traces_file}")
    print(f"{'='*60}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print("\nDone! The traces are compatible with pilot1/pilot2/pilot3 scripts.")
    print("Run analyses with:")
    print(f"  python3 pilot1_transitions.py --model {MODEL_NAME}")
    print(f"  python3 pilot2_dynamics.py --model {MODEL_NAME}")
    print(f"  python3 pilot3_markers.py --model {MODEL_NAME}")


if __name__ == "__main__":
    main()
