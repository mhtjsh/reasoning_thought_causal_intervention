"""
Generate reasoning traces using plain HuggingFace transformers.
NO vLLM dependency. Processes questions one-by-one for robustness.

Usage:
    python generate_traces.py                          # All models, 20 questions
    python generate_traces.py --model qwq-32b          # Single model
    python generate_traces.py --model qwen3-8b --n 5   # 5 questions only
"""

import argparse
import json
import time
import sys
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    REASONING_MODELS,
    TRACES_DIR,
    MEDQA_PROMPT_TEMPLATE,
    ReasoningModelConfig,
    extract_thinking_and_answer,
    extract_predicted_option,
    remove_repetitions,
)
from datasets_medqa import load_medqa, select_pilot_sample, load_from_jsonl, save_to_jsonl


@dataclass
class ReasoningTrace:
    """A single reasoning trace."""
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

    @classmethod
    def from_dict(cls, d: dict) -> "ReasoningTrace":
        # Handle traces that were saved without some fields
        d.setdefault("thinking_char_count", len(d.get("thinking_text", "")))
        d.setdefault("output_tokens", 0)
        return cls(**d)


def load_model_and_tokenizer(config: ReasoningModelConfig):
    """
    Load model and tokenizer with robust error handling.

    Uses device_map="auto" for automatic GPU distribution.
    Falls back to 4-bit quantization if OOM.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Loading: {config.hf_id}")
    print(f"  device_map: {config.device_map}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_id,
        trust_remote_code=True,
        padding_side="left",
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_id,
            torch_dtype=torch.bfloat16,
            device_map=config.device_map,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    except Exception as e:
        print(f"  ⚠ Flash attention failed ({e}), trying without it...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.hf_id,
                torch_dtype=torch.bfloat16,
                device_map=config.device_map,
                trust_remote_code=True,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠ OOM! Trying 4-bit quantization...")
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                config.hf_id,
                quantization_config=quantization_config,
                device_map=config.device_map,
                trust_remote_code=True,
            )

    model.eval()
    print(f"  ✓ Model loaded successfully")
    return model, tokenizer


def build_prompt(question, tokenizer, config: ReasoningModelConfig) -> str:
    """Build the prompt with proper chat template."""
    prompt_text = MEDQA_PROMPT_TEMPLATE.format(
        clinical_vignette=question.clinical_vignette,
        options_text=question.options_text,
    )

    messages = [{"role": "user", "content": prompt_text}]

    try:
        # Use chat template if available
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if config.enable_thinking:
            kwargs["enable_thinking"] = True

        formatted = tokenizer.apply_chat_template(messages, **kwargs)
        return formatted
    except Exception as e:
        # Fallback: raw prompt
        print(f"  ⚠ Chat template failed ({e}), using raw prompt")
        return prompt_text


def generate_single(
    model,
    tokenizer,
    question,
    config: ReasoningModelConfig,
) -> ReasoningTrace:
    """Generate a trace for a single question with robust error handling."""
    prompt = build_prompt(question, tokenizer, config)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # Generate
    start = time.time()
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                do_sample=True,
            )
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        # Retry with fewer tokens
        print(f"    ⚠ OOM on {question.question_id}, retrying with 2048 tokens...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                do_sample=True,
            )
    gen_time = time.time() - start

    # Decode only new tokens
    new_tokens = outputs[0][input_len:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    output_tokens = len(new_tokens)

    # Extract thinking and answer
    thinking, answer = extract_thinking_and_answer(raw_output, config.has_think_tags)
    predicted = extract_predicted_option(answer)

    # If answer extraction from answer section failed, try from thinking
    if predicted is None:
        predicted = extract_predicted_option(thinking)

    trace = ReasoningTrace(
        question_id=question.question_id,
        model_name=config.name,
        raw_output=remove_repetitions(raw_output),
        thinking_text=thinking,
        final_answer=answer[:2000],  # Truncate huge answer sections
        predicted_option=predicted,
        correct_option=question.correct_answer,
        is_correct=(predicted == question.correct_answer) if predicted else False,
        generation_time_s=gen_time,
        output_tokens=output_tokens,
        thinking_char_count=len(thinking),
    )

    status = "✓" if trace.is_correct else "✗"
    parsed = trace.predicted_option or "?"
    print(f"    [{status}] {question.question_id}: "
          f"predicted={parsed} correct={question.correct_answer} "
          f"({gen_time:.1f}s, {output_tokens} tokens, "
          f"{len(thinking)} chars thinking)")

    return trace


def generate_traces_for_model(
    config: ReasoningModelConfig,
    questions: list,
) -> List[ReasoningTrace]:
    """Generate traces for all questions using one model."""
    output_dir = TRACES_DIR / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_file = output_dir / "traces.jsonl"

    # Check for existing traces (resume support)
    existing = {}
    if traces_file.exists():
        with open(traces_file, "r") as f:
            for line in f:
                d = json.loads(line.strip())
                existing[d["question_id"]] = d
        print(f"  Found {len(existing)} existing traces, will skip those")

    # Load model
    model, tokenizer = load_model_and_tokenizer(config)

    traces = []
    for i, q in enumerate(questions):
        if q.question_id in existing:
            traces.append(ReasoningTrace.from_dict(existing[q.question_id]))
            print(f"  [{i+1}/{len(questions)}] {q.question_id}: SKIPPED (cached)")
            continue

        print(f"  [{i+1}/{len(questions)}] Generating...")
        try:
            trace = generate_single(model, tokenizer, q, config)
            traces.append(trace)

            # Save incrementally (so we don't lose progress on crashes)
            with open(traces_file, "a") as f:
                f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"    ✗ ERROR on {q.question_id}: {e}")
            # Save a placeholder trace
            trace = ReasoningTrace(
                question_id=q.question_id,
                model_name=config.name,
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

    # Print summary
    n_correct = sum(1 for t in traces if t.is_correct)
    n_parsed = sum(1 for t in traces if t.predicted_option)
    n_errors = sum(1 for t in traces if t.raw_output.startswith("ERROR"))
    avg_thinking = sum(t.thinking_char_count for t in traces) / max(len(traces), 1)

    print(f"\n  RESULTS for {config.name}:")
    print(f"    Accuracy:       {n_correct}/{len(traces)} ({100*n_correct/len(traces):.1f}%)")
    print(f"    Parsed answers: {n_parsed}/{len(traces)}")
    print(f"    Errors:         {n_errors}")
    print(f"    Avg thinking:   {avg_thinking:.0f} chars")
    print(f"    Saved to:       {traces_file}")

    # Free GPU memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return traces


def load_traces(model_name: str) -> List[ReasoningTrace]:
    """Load saved traces from disk."""
    traces_file = TRACES_DIR / model_name / "traces.jsonl"
    if not traces_file.exists():
        raise FileNotFoundError(f"No traces: {traces_file}")

    traces = []
    with open(traces_file, "r") as f:
        for line in f:
            if line.strip():
                traces.append(ReasoningTrace.from_dict(json.loads(line.strip())))
    print(f"Loaded {len(traces)} traces for {model_name}")
    return traces


def main():
    parser = argparse.ArgumentParser(description="Generate reasoning traces")
    parser.add_argument("--model", type=str, default=None,
                       choices=list(REASONING_MODELS.keys()),
                       help="Single model to run (default: all)")
    parser.add_argument("--n", type=int, default=20,
                       help="Number of pilot questions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load questions
    from config import MEDQA_DIR
    pilot_file = MEDQA_DIR / "pilot_sample.jsonl"
    if pilot_file.exists():
        questions = load_from_jsonl(pilot_file)
        print(f"Loaded {len(questions)} pilot questions from cache")
    else:
        all_qs = load_medqa()
        questions = select_pilot_sample(all_qs, n=args.n, seed=args.seed)
        save_to_jsonl(questions, pilot_file)

    # Limit if requested
    if args.n < len(questions):
        questions = questions[:args.n]

    # Which models to run
    if args.model:
        model_names = [args.model]
    else:
        model_names = list(REASONING_MODELS.keys())

    print(f"\nModels: {model_names}")
    print(f"Questions: {len(questions)}")

    # Generate sequentially (one model at a time)
    for name in model_names:
        config = REASONING_MODELS[name]
        print(f"\n{'='*60}")
        print(f"MODEL: {config.name} ({config.hf_id})")
        print(f"{'='*60}")
        generate_traces_for_model(config, questions)

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
