"""
Pilot 2: Do embedding dynamics differ for correct vs. incorrect traces?

Tests whether measurable features of the cosine similarity curve
are statistically different between correct and incorrect answers.

Features aligned with the paper plan:
- Number of hypothesis transitions
- Mean/min/std consecutive similarity
- Similarity slope (convergence rate)
- Midpoint similarity (how committed at halfway)
- Max drop magnitude (sharpest hypothesis shift)
- Final convergence (last 3 sentences)
- Commitment Signature C (first position where similarity stays above τ)
- Normalized commitment (C / trace_length — early = premature)

Uses Mann-Whitney U (non-parametric) and Cohen's d (effect size).

Usage:
    python pilot2_dynamics.py                        # All models
    python pilot2_dynamics.py --model qwq-32b        # Single model
"""

import argparse
import json
import sys
import gc
import numpy as np
from pathlib import Path
from typing import List, Dict

import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    REASONING_MODELS,
    EMBEDDING_MODELS,
    RESULTS_DIR,
    PILOT_CONFIG,
)
from generate_traces import load_traces
from pilot1_transitions import (
    split_into_sentences,
    load_embedding_model,
    encode_sentences,
    compute_similarity_curve,
    detect_transitions,
)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled)


def mannwhitney_u(group1: np.ndarray, group2: np.ndarray) -> float:
    """Mann-Whitney U test p-value. Returns 1.0 if insufficient data."""
    if len(group1) < 2 or len(group2) < 2:
        return 1.0
    try:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(group1, group2, alternative="two-sided")
        return float(p)
    except (ValueError, ImportError):
        return 1.0


def compute_commitment_signature(
    similarities: List[float],
    tau: float = 0.8,
    min_sustain: int = 2,
) -> int:
    """
    Compute Commitment Signature C — the first depth position where
    consecutive similarity stays above threshold τ for `min_sustain`
    consecutive steps.

    Returns:
        Position index of commitment, or -1 if never committed.
    """
    sims = similarities
    for i in range(len(sims)):
        if sims[i] >= tau:
            # Check if it stays above τ for min_sustain steps
            stays = True
            for j in range(i, min(i + min_sustain, len(sims))):
                if sims[j] < tau:
                    stays = False
                    break
            if stays:
                return i
    return -1  # Never committed


def compute_trace_features(
    trace,
    emb_model,
    drop_threshold: float,
) -> Dict:
    """
    Compute feature vector for a single trace.

    Features (aligned with paper plan's "Metrics to compute per trace"):
    - n_transitions: number of detected hypothesis transitions
    - slope: linear slope of similarity curve (pos = convergence)
    - mean_sim: average consecutive similarity
    - min_sim: minimum consecutive similarity
    - std_sim: standard deviation of similarity
    - trace_length: number of sentences
    - midpoint_sim: similarity at the halfway point
    - max_drop: largest single-step similarity decrease
    - final_convergence: mean similarity of last 3 pairs
    - commitment_point: Commitment Signature C (position)
    - normalized_commitment: C / trace_length (0-1, lower = earlier commit)
    """
    sentences = split_into_sentences(trace.thinking_text)
    if len(sentences) < 3:
        return None

    embeddings = encode_sentences(emb_model, sentences)
    if len(embeddings) < 3:
        return None

    sims = compute_similarity_curve(embeddings)
    transitions = detect_transitions(sims, drop_threshold)

    x = np.arange(len(sims))
    slope = float(np.polyfit(x, sims, 1)[0]) if len(sims) >= 2 else 0.0

    # Midpoint similarity (how committed at halfway)
    midpoint = len(sims) // 2
    midpoint_sim = float(sims[midpoint]) if midpoint < len(sims) else float(np.mean(sims))

    # Max drop magnitude (sharpest single hypothesis shift)
    drops = [sims[i] - sims[i + 1] for i in range(len(sims) - 1)]
    max_drop = float(max(drops)) if drops else 0.0

    # Final convergence (mean of last 3 similarity values)
    last_n = min(3, len(sims))
    final_convergence = float(np.mean(sims[-last_n:]))

    # Commitment Signature C (try multiple τ values for robustness)
    commitment_point = compute_commitment_signature(sims, tau=0.8)
    normalized_commitment = (
        commitment_point / len(sims) if commitment_point >= 0 else 1.0
    )

    # Drift-to-conclusion: similarity of each step's embedding to the final embedding
    drift_to_final = [
        float(np.dot(embeddings[i], embeddings[-1]) /
              (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[-1]) + 1e-9))
        for i in range(len(embeddings) - 1)
    ]
    mean_drift = float(np.mean(drift_to_final)) if drift_to_final else 0.0

    return {
        "question_id": trace.question_id,
        "is_correct": trace.is_correct,
        "n_transitions": len(transitions),
        "slope": slope,
        "mean_sim": float(np.mean(sims)),
        "min_sim": float(np.min(sims)),
        "std_sim": float(np.std(sims)),
        "trace_length": len(sentences),
        "thinking_chars": len(trace.thinking_text),
        "midpoint_sim": midpoint_sim,
        "max_drop": max_drop,
        "final_convergence": final_convergence,
        "commitment_point": commitment_point,
        "normalized_commitment": normalized_commitment,
        "mean_drift_to_final": mean_drift,
        "similarities": sims,
        "drift_to_final": drift_to_final,
    }


def run_pilot2(model_names: List[str], embedding_names: List[str]) -> Dict:
    """
    Run Pilot 2: Statistical comparison of correct vs incorrect dynamics.
    """
    results = {}
    feature_names = [
        "n_transitions", "slope", "mean_sim", "min_sim", "std_sim",
        "trace_length", "midpoint_sim", "max_drop", "final_convergence",
        "commitment_point", "normalized_commitment", "mean_drift_to_final",
    ]

    for emb_name in embedding_names:
        emb_config = EMBEDDING_MODELS[emb_name]
        print(f"\n{'='*60}")
        print(f"EMBEDDING: {emb_name}")
        print(f"{'='*60}")

        emb_model = load_embedding_model(emb_config)

        for model_name in model_names:
            try:
                traces = load_traces(model_name)
            except FileNotFoundError:
                print(f"  ⚠ No traces for {model_name}")
                continue

            if len(traces) < 5:
                print(f"  ⚠ Too few traces ({len(traces)})")
                continue

            key = f"{model_name}__{emb_name}"
            print(f"\n  Model: {model_name}")

            # Compute features for each trace
            features_list = []
            for trace in traces:
                if not trace.thinking_text or len(trace.thinking_text) < 50:
                    continue
                feat = compute_trace_features(
                    trace, emb_model, PILOT_CONFIG.cosine_drop_threshold
                )
                if feat:
                    features_list.append(feat)

            if len(features_list) < 5:
                print(f"    Insufficient features ({len(features_list)})")
                continue

            # Split into correct vs incorrect
            correct = [f for f in features_list if f["is_correct"]]
            incorrect = [f for f in features_list if not f["is_correct"]]

            print(f"    Correct: {len(correct)}, Incorrect: {len(incorrect)}")

            if len(correct) < 2 or len(incorrect) < 2:
                print(f"    ⚠ Too few in one group")
                results[key] = {
                    "reasoning_model": model_name,
                    "embedding_model": emb_name,
                    "n_correct": len(correct),
                    "n_incorrect": len(incorrect),
                    "warning": "insufficient_group_size",
                }
                continue

            # Statistical tests per feature
            stats = {}
            for feat in feature_names:
                c = np.array([f[feat] for f in correct if f[feat] is not None and f[feat] != -1])
                w = np.array([f[feat] for f in incorrect if f[feat] is not None and f[feat] != -1])

                if len(c) < 2 or len(w) < 2:
                    stats[feat] = {
                        "correct_mean": float(np.mean(c)) if len(c) > 0 else 0.0,
                        "incorrect_mean": float(np.mean(w)) if len(w) > 0 else 0.0,
                        "mann_whitney_p": 1.0,
                        "cohens_d": 0.0,
                        "significant": False,
                        "note": "insufficient_data",
                    }
                    continue

                p_value = mannwhitney_u(c, w)
                d = cohens_d(c, w)

                stats[feat] = {
                    "correct_mean": float(np.mean(c)),
                    "correct_std": float(np.std(c)),
                    "incorrect_mean": float(np.mean(w)),
                    "incorrect_std": float(np.std(w)),
                    "mann_whitney_p": p_value,
                    "cohens_d": d,
                    "significant": p_value < PILOT_CONFIG.pilot2_p_value_threshold,
                }

                sig = "✓" if stats[feat]["significant"] else "✗"
                print(f"    [{sig}] {feat:25s}: "
                      f"correct={stats[feat]['correct_mean']:.3f}±{stats[feat].get('correct_std',0):.3f}, "
                      f"incorrect={stats[feat]['incorrect_mean']:.3f}±{stats[feat].get('incorrect_std',0):.3f}, "
                      f"p={p_value:.4f}, d={d:.3f}")

            results[key] = {
                "reasoning_model": model_name,
                "embedding_model": emb_name,
                "n_correct": len(correct),
                "n_incorrect": len(incorrect),
                "n_total": len(features_list),
                "stats": stats,
                "n_significant_features": sum(1 for s in stats.values() if s.get("significant", False)),
                "all_features": features_list,
            }

        # Cleanup
        emb_model.cleanup()

    return results


def save_results(results: Dict, filename: str = "pilot2_results.json"):
    output = RESULTS_DIR / filename
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    clean = json.loads(json.dumps(results, default=convert))

    # Merge with existing file if present
    if output.exists():
        try:
            with open(output) as f:
                existing = json.load(f)
            existing.update(clean)
            clean = existing
        except (json.JSONDecodeError, KeyError):
            pass

    with open(output, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n✓ Results → {output}")


def main():
    parser = argparse.ArgumentParser(description="Pilot 2: Dynamics Predict Correctness")
    parser.add_argument("--model", type=str, default=None,
                       choices=list(REASONING_MODELS.keys()))
    parser.add_argument("--embedding", type=str, default=None,
                       choices=list(EMBEDDING_MODELS.keys()))
    args = parser.parse_args()

    models = [args.model] if args.model else list(REASONING_MODELS.keys())
    embeddings = [args.embedding] if args.embedding else list(EMBEDDING_MODELS.keys())

    print("="*60)
    print("PILOT 2: Embedding Dynamics Predict Correctness")
    print("="*60)

    results = run_pilot2(models, embeddings)
    save_results(results)

    # Summary
    print("\n" + "="*60)
    print("PILOT 2 SUMMARY")
    print("="*60)
    for key, data in results.items():
        n_sig = data.get("n_significant_features", 0)
        warn = data.get("warning", "")
        total_feats = len([s for s in data.get("stats", {}).values()
                          if "note" not in s])
        status = f"⚠ {warn}" if warn else f"{n_sig}/{total_feats} features significant"
        print(f"  {data.get('reasoning_model','?')} + {data.get('embedding_model','?')}: {status}")


if __name__ == "__main__":
    main()
