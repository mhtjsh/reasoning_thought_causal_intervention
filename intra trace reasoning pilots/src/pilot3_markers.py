"""
Pilot 3: Are uncertainty markers calibrated with errors?

Tests whether the frequency of hedging, revision, verification,
and explicit uncertainty markers in reasoning traces correlates
with model error rate.

Improvements:
- Uses correct Spearman rho threshold (not p-value) for pass/fail
- Adds per-position analysis (early vs late trace regions)
- Adds per-category breakdown report

Usage:
    python pilot3_markers.py                        # All models
    python pilot3_markers.py --model qwq-32b        # Single model
"""

import argparse
import json
import re
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    REASONING_MODELS,
    RESULTS_DIR,
    PILOT_CONFIG,
    UNCERTAINTY_MARKERS,
)
from generate_traces import load_traces


def count_markers(text: str) -> Dict[str, int]:
    """Count occurrences of each marker category."""
    if not text:
        return {cat: 0 for cat in UNCERTAINTY_MARKERS}

    counts = {}
    for category, patterns in UNCERTAINTY_MARKERS.items():
        n = 0
        for pattern in patterns:
            n += len(re.findall(pattern, text, re.IGNORECASE))
        counts[category] = n
    return counts


def count_markers_in_region(text: str, start_frac: float, end_frac: float) -> Dict[str, int]:
    """Count markers in a specific region of the text (by character fraction)."""
    if not text:
        return {cat: 0 for cat in UNCERTAINTY_MARKERS}
    start = int(len(text) * start_frac)
    end = int(len(text) * end_frac)
    region = text[start:end]
    return count_markers(region)


def normalize_markers(counts: Dict[str, int], text_length: int) -> Dict[str, float]:
    """Normalize marker counts per 1000 characters."""
    if text_length < 1:
        return {cat: 0.0 for cat in counts}
    return {cat: count * 1000 / text_length for cat, count in counts.items()}


def spearman_corr(x: np.ndarray, y: np.ndarray) -> tuple:
    """Spearman rank correlation. Returns (rho, p_value)."""
    if len(x) < 3 or len(y) < 3:
        return 0.0, 1.0
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(x, y)
        if np.isnan(rho):
            return 0.0, 1.0
        return float(rho), float(p)
    except (ValueError, ImportError):
        return 0.0, 1.0


def run_pilot3(model_names: List[str]) -> Dict:
    """
    Run Pilot 3: Uncertainty Marker Calibration.

    For each model:
    1. Count markers in thinking text (overall + early/late halves)
    2. Correlate marker density with error (1-is_correct)
    3. Check if Spearman rho > threshold (not just p-value)
    """
    results = {}
    categories = list(UNCERTAINTY_MARKERS.keys())

    for model_name in model_names:
        try:
            traces = load_traces(model_name)
        except FileNotFoundError:
            print(f"  ⚠ No traces for {model_name}")
            continue

        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        # Filter to traces with actual thinking
        valid = [t for t in traces if t.thinking_text and len(t.thinking_text) > 50]
        if len(valid) < 5:
            print(f"  Too few valid traces ({len(valid)})")
            continue

        # Compute markers for each trace
        trace_data = []
        for trace in valid:
            counts = count_markers(trace.thinking_text)
            normalized = normalize_markers(counts, len(trace.thinking_text))
            total = sum(counts.values())
            total_norm = sum(normalized.values())

            # Per-position analysis: early half vs late half
            early_counts = count_markers_in_region(trace.thinking_text, 0.0, 0.5)
            late_counts = count_markers_in_region(trace.thinking_text, 0.5, 1.0)
            half_len = max(len(trace.thinking_text) // 2, 1)
            early_norm = normalize_markers(early_counts, half_len)
            late_norm = normalize_markers(late_counts, half_len)

            trace_data.append({
                "question_id": trace.question_id,
                "is_correct": trace.is_correct,
                "is_error": not trace.is_correct,
                "thinking_chars": len(trace.thinking_text),
                "raw_counts": counts,
                "normalized_counts": normalized,
                "total_markers": total,
                "total_markers_normalized": total_norm,
                "early_normalized": early_norm,
                "late_normalized": late_norm,
            })

        # Binary error variable (1 = wrong, 0 = correct)
        errors = np.array([1 if d["is_error"] else 0 for d in trace_data])

        # Correlate each category with error
        correlations = {}
        for cat in categories:
            densities = np.array([d["normalized_counts"][cat] for d in trace_data])
            rho, p = spearman_corr(densities, errors)

            correct_density = np.mean([d["normalized_counts"][cat]
                                       for d in trace_data if d["is_correct"]])
            incorrect_density = np.mean([d["normalized_counts"][cat]
                                         for d in trace_data if d["is_error"]])

            # Pass criteria: Spearman rho > threshold AND p < 0.05
            passes_rho = abs(rho) > PILOT_CONFIG.pilot3_spearman_threshold
            passes_p = p < PILOT_CONFIG.pilot2_p_value_threshold

            correlations[cat] = {
                "spearman_rho": rho,
                "p_value": p,
                "passes_rho_threshold": passes_rho,
                "passes_p_value": passes_p,
                "significant": passes_rho,  # Primary criterion: |rho| > 0.15
                "mean_density_correct": float(correct_density) if not np.isnan(correct_density) else 0.0,
                "mean_density_incorrect": float(incorrect_density) if not np.isnan(incorrect_density) else 0.0,
            }

            sig = "✓" if correlations[cat]["significant"] else "✗"
            print(f"  [{sig}] {cat:10s}: "
                  f"ρ={rho:+.3f}, p={p:.4f}, "
                  f"correct={correct_density:.2f}/k, "
                  f"incorrect={incorrect_density:.2f}/k")

        # Total markers correlation
        total_densities = np.array([d["total_markers_normalized"] for d in trace_data])
        total_rho, total_p = spearman_corr(total_densities, errors)
        print(f"  [{'✓' if abs(total_rho) > 0.15 else '✗'}] {'TOTAL':10s}: "
              f"ρ={total_rho:+.3f}, p={total_p:.4f}")

        # Per-position analysis: are markers more informative in early vs late?
        print(f"\n  Position analysis (early half vs late half):")
        for cat in categories:
            early_densities = np.array([d["early_normalized"][cat] for d in trace_data])
            late_densities = np.array([d["late_normalized"][cat] for d in trace_data])
            early_rho, _ = spearman_corr(early_densities, errors)
            late_rho, _ = spearman_corr(late_densities, errors)
            better = "EARLY" if abs(early_rho) > abs(late_rho) else "LATE"
            print(f"    {cat:10s}: early ρ={early_rho:+.3f}, late ρ={late_rho:+.3f} → {better}")

        # Summary stats
        n_correct = sum(1 for d in trace_data if d["is_correct"])
        n_incorrect = len(trace_data) - n_correct

        results[model_name] = {
            "n_traces": len(trace_data),
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "accuracy": n_correct / len(trace_data),
            "correlations": correlations,
            "total_rho": total_rho,
            "total_p": total_p,
            "n_significant_categories": sum(
                1 for c in correlations.values() if c["significant"]
            ),
            "top_markers_in_errors": {
                cat: float(np.mean([d["normalized_counts"][cat]
                                    for d in trace_data if d["is_error"]]))
                for cat in categories
            } if n_incorrect > 0 else {},
            "trace_data": trace_data,
        }

    return results


def save_results(results: Dict, filename: str = "pilot3_results.json"):
    output = RESULTS_DIR / filename
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    clean = json.loads(json.dumps(results, default=convert))
    with open(output, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n✓ Results → {output}")


def generate_text_report(results: Dict) -> str:
    """Generate a human-readable summary report."""
    lines = []
    lines.append("=" * 60)
    lines.append("PILOT 3: Uncertainty Marker Calibration — Report")
    lines.append("=" * 60)

    for model_name, data in results.items():
        lines.append(f"\n{'─' * 40}")
        lines.append(f"Model: {model_name}")
        lines.append(f"  Accuracy: {data['accuracy']:.1%} "
                     f"({data['n_correct']}/{data['n_traces']})")
        lines.append(f"  Significant categories (|ρ| > {PILOT_CONFIG.pilot3_spearman_threshold}): "
                     f"{data['n_significant_categories']}/4")

        for cat, corr in data["correlations"].items():
            sig = "✓" if corr["significant"] else "✗"
            lines.append(f"  [{sig}] {cat}: ρ={corr['spearman_rho']:+.3f}, "
                        f"p={corr['p_value']:.4f}")

        lines.append(f"  Total markers: ρ={data['total_rho']:+.3f}, "
                     f"p={data['total_p']:.4f}")

    # Decision
    lines.append(f"\n{'=' * 60}")
    lines.append("PASS/FAIL CRITERIA:")
    lines.append(f"  Criterion: |ρ| > {PILOT_CONFIG.pilot3_spearman_threshold} for ≥1 category")
    n_passing = sum(
        1 for d in results.values()
        if d["n_significant_categories"] >= 1 or abs(d["total_rho"]) > PILOT_CONFIG.pilot3_spearman_threshold
    )
    threshold = PILOT_CONFIG.pilot3_min_models_passing
    passed = n_passing >= threshold
    lines.append(f"  Models with signal: {n_passing}")
    lines.append(f"  Required: {threshold}")
    lines.append(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    if not passed:
        lines.append(f"\n  NOTE: If markers fail, the plan recommends dropping markers")
        lines.append(f"  and diagnosing from embedding dynamics (Commitment Signature C) alone.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Pilot 3: Uncertainty Markers")
    parser.add_argument("--model", type=str, default=None,
                       choices=list(REASONING_MODELS.keys()))
    args = parser.parse_args()

    models = [args.model] if args.model else list(REASONING_MODELS.keys())

    print("="*60)
    print("PILOT 3: Uncertainty Marker Calibration")
    print("="*60)

    results = run_pilot3(models)
    save_results(results)

    report = generate_text_report(results)
    print(report)

    # Save text report
    report_file = RESULTS_DIR / "pilot3_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"\n✓ Report → {report_file}")


if __name__ == "__main__":
    main()
