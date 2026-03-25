"""
Orchestrator: Run the full pilot experiment pipeline.

Usage:
    python run_pilots.py                          # Full pipeline
    python run_pilots.py --step pilot1            # Single step
    python run_pilots.py --step pilot3 --model qwq-32b
    python run_pilots.py --step summary           # Generate summary
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import REASONING_MODELS, EMBEDDING_MODELS, RESULTS_DIR, TRACES_DIR


def run_download():
    """Download MedQA data."""
    print("\n[STEP] Download MedQA")
    from datasets_medqa import load_medqa, select_pilot_sample, save_to_jsonl
    from config import MEDQA_DIR

    pilot_file = MEDQA_DIR / "pilot_sample.jsonl"
    if pilot_file.exists():
        print(f"  Already cached: {pilot_file}")
        return

    all_qs = load_medqa()
    pilot = select_pilot_sample(all_qs, n=20, seed=42)
    save_to_jsonl(pilot, pilot_file)
    print(f"  Saved {len(pilot)} pilot questions")


def run_generate(model_names):
    """Generate traces for specified models."""
    print("\n[STEP] Generate Traces")
    for model_name in model_names:
        trace_file = TRACES_DIR / model_name / "traces.jsonl"
        if trace_file.exists():
            import json
            with open(trace_file) as f:
                n = sum(1 for _ in f)
            print(f"  {model_name}: {n} traces (cached)")
            continue

        if model_name == "qwen3.5-27b":
            print(f"  {model_name}: Run generate_traces_qwen35.py separately")
            print(f"    → python src/generate_traces_qwen35.py")
        else:
            print(f"  {model_name}: Run generate_traces.py --model {model_name}")
            print(f"    → python src/generate_traces.py --model {model_name}")


def run_pilot1(model_names, embedding_names):
    """Run Pilot 1: Hypothesis Transition Detection."""
    print("\n[STEP] Pilot 1: Hypothesis Transitions")
    from pilot1_transitions import run_pilot1 as _run, save_results

    # Filter to models with existing traces
    available = [m for m in model_names if (TRACES_DIR / m / "traces.jsonl").exists()]
    missing = set(model_names) - set(available)
    if missing:
        print(f"  ⚠ Skipping (no traces): {missing}")

    if not available:
        print("  ⚠ No traces available, skipping Pilot 1")
        return

    results = _run(available, embedding_names)
    save_results(results)


def run_pilot2(model_names, embedding_names):
    """Run Pilot 2: Dynamics Predict Correctness."""
    print("\n[STEP] Pilot 2: Embedding Dynamics")
    from pilot2_dynamics import run_pilot2 as _run, save_results

    available = [m for m in model_names if (TRACES_DIR / m / "traces.jsonl").exists()]
    missing = set(model_names) - set(available)
    if missing:
        print(f"  ⚠ Skipping (no traces): {missing}")

    if not available:
        print("  ⚠ No traces available, skipping Pilot 2")
        return

    results = _run(available, embedding_names)
    save_results(results)


def run_pilot3(model_names):
    """Run Pilot 3: Uncertainty Markers."""
    print("\n[STEP] Pilot 3: Uncertainty Markers")
    from pilot3_markers import run_pilot3 as _run, save_results

    available = [m for m in model_names if (TRACES_DIR / m / "traces.jsonl").exists()]
    missing = set(model_names) - set(available)
    if missing:
        print(f"  ⚠ Skipping (no traces): {missing}")

    if not available:
        print("  ⚠ No traces available, skipping Pilot 3")
        return

    results = _run(available)
    save_results(results)


def run_summary():
    """Generate consolidated summary report."""
    print("\n[STEP] Pilot Summary")
    from pilot_summary import main as summary_main
    summary_main()


def main():
    parser = argparse.ArgumentParser(
        description="Run pilot experiments pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
    download    Download MedQA dataset
    generate    Generate traces (shows instructions)
    pilot1      Run Pilot 1 (hypothesis transitions)
    pilot2      Run Pilot 2 (dynamics predict correctness)
    pilot3      Run Pilot 3 (uncertainty markers)
    summary     Generate summary report
    all         Run everything (default)
        """,
    )
    parser.add_argument(
        "--step", type=str, default="all",
        choices=["download", "generate", "pilot1", "pilot2", "pilot3", "summary", "all"],
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(REASONING_MODELS.keys()),
    )
    parser.add_argument(
        "--embedding", type=str, default=None,
        choices=list(EMBEDDING_MODELS.keys()),
    )
    args = parser.parse_args()

    models = [args.model] if args.model else list(REASONING_MODELS.keys())
    embeddings = [args.embedding] if args.embedding else list(EMBEDDING_MODELS.keys())

    print("="*60)
    print("PILOT EXPERIMENT PIPELINE")
    print("="*60)
    print(f"  Step: {args.step}")
    print(f"  Models: {models}")
    print(f"  Embeddings: {embeddings}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.step in ("download", "all"):
        run_download()

    if args.step in ("generate", "all"):
        run_generate(models)

    if args.step in ("pilot1", "all"):
        run_pilot1(models, embeddings)

    if args.step in ("pilot2", "all"):
        run_pilot2(models, embeddings)

    if args.step in ("pilot3", "all"):
        run_pilot3(models)

    if args.step in ("summary", "all"):
        run_summary()

    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == "__main__":
    main()
