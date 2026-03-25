"""
Pilot Summary: Consolidated report of all pilot experiment results.

Generates a markdown report with:
- Pass/fail table for each pilot × model × embedding
- Similarity curve visualization (hero figure)
- Recommendations for which models/embeddings to carry forward

Usage:
    python pilot_summary.py  # After running all pilots
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, PILOT_CONFIG, REASONING_MODELS, EMBEDDING_MODELS


def load_json(filename: str) -> Optional[Dict]:
    """Load a JSON results file, return None if not found."""
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"  ⚠ Not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def generate_similarity_plots(pilot2_data: Dict) -> List[str]:
    """Generate similarity curve plots colored by correctness."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib not available, skipping plots")
        return []

    plot_paths = []

    for key, data in pilot2_data.items():
        if "all_features" not in data:
            continue

        model_name = data.get("reasoning_model", "unknown")
        emb_name = data.get("embedding_model", "unknown")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Intra-Trace Dynamics: {model_name} + {emb_name}",
            fontsize=14, fontweight="bold"
        )

        correct_features = [f for f in data["all_features"] if f["is_correct"]]
        incorrect_features = [f for f in data["all_features"] if not f["is_correct"]]

        # Plot 1: Similarity curves
        ax = axes[0]
        for f in correct_features:
            sims = f.get("similarities", [])
            if sims:
                ax.plot(sims, color="steelblue", alpha=0.4, linewidth=1)
        for f in incorrect_features:
            sims = f.get("similarities", [])
            if sims:
                ax.plot(sims, color="indianred", alpha=0.4, linewidth=1)

        # Plot mean curves
        if correct_features:
            max_len = max(len(f.get("similarities", [])) for f in correct_features)
            if max_len > 0:
                padded = []
                for f in correct_features:
                    s = f.get("similarities", [])
                    if len(s) > 0:
                        padded.append(np.interp(
                            np.linspace(0, 1, max_len),
                            np.linspace(0, 1, len(s)),
                            s
                        ))
                if padded:
                    mean_correct = np.mean(padded, axis=0)
                    ax.plot(mean_correct, color="steelblue", linewidth=3, label="Correct (mean)")

        if incorrect_features:
            max_len = max(len(f.get("similarities", [])) for f in incorrect_features)
            if max_len > 0:
                padded = []
                for f in incorrect_features:
                    s = f.get("similarities", [])
                    if len(s) > 0:
                        padded.append(np.interp(
                            np.linspace(0, 1, max_len),
                            np.linspace(0, 1, len(s)),
                            s
                        ))
                if padded:
                    mean_incorrect = np.mean(padded, axis=0)
                    ax.plot(mean_incorrect, color="indianred", linewidth=3, label="Incorrect (mean)")

        ax.set_xlabel("Step (depth in trace)")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Consecutive Similarity Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Drift to final answer
        ax = axes[1]
        for f in correct_features:
            drift = f.get("drift_to_final", [])
            if drift:
                ax.plot(drift, color="steelblue", alpha=0.4, linewidth=1)
        for f in incorrect_features:
            drift = f.get("drift_to_final", [])
            if drift:
                ax.plot(drift, color="indianred", alpha=0.4, linewidth=1)

        ax.set_xlabel("Step (depth in trace)")
        ax.set_ylabel("Cosine Similarity to Final Answer")
        ax.set_title("Drift-to-Conclusion")
        ax.grid(True, alpha=0.3)

        # Plot 3: Feature comparison (box plots)
        ax = axes[2]
        feat_names = ["n_transitions", "mean_sim", "normalized_commitment"]
        positions = np.arange(len(feat_names))
        width = 0.35

        correct_vals = []
        incorrect_vals = []
        for feat in feat_names:
            c_vals = [f[feat] for f in correct_features if f.get(feat) is not None and f.get(feat) != -1]
            i_vals = [f[feat] for f in incorrect_features if f.get(feat) is not None and f.get(feat) != -1]
            correct_vals.append(c_vals if c_vals else [0])
            incorrect_vals.append(i_vals if i_vals else [0])

        bp1 = ax.boxplot(correct_vals, positions=positions - width/2,
                         widths=width, patch_artist=True,
                         boxprops=dict(facecolor="steelblue", alpha=0.7))
        bp2 = ax.boxplot(incorrect_vals, positions=positions + width/2,
                         widths=width, patch_artist=True,
                         boxprops=dict(facecolor="indianred", alpha=0.7))

        ax.set_xticks(positions)
        ax.set_xticklabels(["# Transitions", "Mean Sim", "Norm. Commitment"], fontsize=9)
        ax.set_title("Feature Comparison")
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Correct", "Incorrect"])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = RESULTS_DIR / f"pilot2_plot_{model_name}_{emb_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_paths.append(str(plot_path))
        print(f"  ✓ Plot saved: {plot_path}")

    return plot_paths


def generate_report(
    pilot1: Optional[Dict],
    pilot2: Optional[Dict],
    pilot3: Optional[Dict],
    plot_paths: List[str],
) -> str:
    """Generate consolidated markdown report."""

    lines = []
    lines.append("# Pilot Experiments — Consolidated Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")

    # Build summary table
    lines.append("| Pilot | Question | Result |")
    lines.append("|-------|----------|--------|")

    # Pilot 1 summary
    if pilot1:
        any_has_transitions = any(
            d.get("pct_with_transitions", 0) > 50
            for d in pilot1.values()
        )
        p1_status = "✅ PASS" if any_has_transitions else "⚠️ MARGINAL"
        lines.append(f"| Pilot 1 | Can embeddings detect hypothesis transitions? | {p1_status} |")
    else:
        lines.append("| Pilot 1 | Can embeddings detect hypothesis transitions? | ❓ Not run |")

    # Pilot 2 summary
    if pilot2:
        any_significant = any(
            d.get("n_significant_features", 0) > 0
            for d in pilot2.values()
            if "warning" not in d
        )
        p2_status = "✅ PASS" if any_significant else "❌ FAIL"
        lines.append(f"| Pilot 2 | Do dynamics predict correctness? | {p2_status} |")
    else:
        lines.append("| Pilot 2 | Do dynamics predict correctness? | ❓ Not run |")

    # Pilot 3 summary
    if pilot3:
        n_models_with_signal = sum(
            1 for d in pilot3.values()
            if d.get("n_significant_categories", 0) >= 1
            or abs(d.get("total_rho", 0)) > PILOT_CONFIG.pilot3_spearman_threshold
        )
        p3_status = "✅ PASS" if n_models_with_signal >= PILOT_CONFIG.pilot3_min_models_passing else "❌ FAIL"
        lines.append(f"| Pilot 3 | Are uncertainty markers calibrated? | {p3_status} |")
    else:
        lines.append("| Pilot 3 | Are uncertainty markers calibrated? | ❓ Not run |")

    lines.append("")

    # Pilot 1 details
    if pilot1:
        lines.append("## Pilot 1: Hypothesis Transition Detection")
        lines.append("")
        lines.append("| Reasoning Model | Embedding | Traces | % With Transitions | Mean Trans/Trace | Mean Similarity |")
        lines.append("|-----------------|-----------|--------|--------------------|------------------|-----------------|")
        for key, data in pilot1.items():
            lines.append(
                f"| {data['reasoning_model']} | {data['embedding_model']} | "
                f"{data['n_traces_analyzed']} | "
                f"{data['pct_with_transitions']:.0f}% | "
                f"{data['mean_transitions_per_trace']:.1f} | "
                f"{data['mean_similarity']:.3f} |"
            )
        lines.append("")

    # Pilot 2 details
    if pilot2:
        lines.append("## Pilot 2: Embedding Dynamics Predict Correctness")
        lines.append("")
        for key, data in pilot2.items():
            if "stats" not in data:
                continue
            model = data["reasoning_model"]
            emb = data["embedding_model"]
            lines.append(f"### {model} + {emb}")
            lines.append(f"Correct: {data['n_correct']}, Incorrect: {data['n_incorrect']}")
            lines.append("")
            lines.append("| Feature | Correct (μ±σ) | Incorrect (μ±σ) | p-value | Cohen's d | Sig? |")
            lines.append("|---------|---------------|-----------------|---------|-----------|------|")
            for feat, stats in data["stats"].items():
                if "note" in stats:
                    continue
                sig = "✅" if stats.get("significant") else "—"
                lines.append(
                    f"| {feat} | "
                    f"{stats.get('correct_mean',0):.3f}±{stats.get('correct_std',0):.3f} | "
                    f"{stats.get('incorrect_mean',0):.3f}±{stats.get('incorrect_std',0):.3f} | "
                    f"{stats.get('mann_whitney_p',1):.4f} | "
                    f"{stats.get('cohens_d',0):.3f} | {sig} |"
                )
            lines.append("")

    # Pilot 3 details
    if pilot3:
        lines.append("## Pilot 3: Uncertainty Marker Calibration")
        lines.append("")
        for model_name, data in pilot3.items():
            lines.append(f"### {model_name}")
            lines.append(f"Accuracy: {data['accuracy']:.1%} ({data['n_correct']}/{data['n_traces']})")
            lines.append("")
            lines.append("| Category | Spearman ρ | p-value | Correct density | Incorrect density | Sig? |")
            lines.append("|----------|-----------|---------|-----------------|-------------------|------|")
            for cat, corr in data.get("correlations", {}).items():
                sig = "✅" if corr.get("significant") else "—"
                lines.append(
                    f"| {cat} | {corr['spearman_rho']:+.3f} | {corr['p_value']:.4f} | "
                    f"{corr['mean_density_correct']:.2f}/k | "
                    f"{corr['mean_density_incorrect']:.2f}/k | {sig} |"
                )
            lines.append(f"\nTotal: ρ={data['total_rho']:+.3f}, p={data['total_p']:.4f}")
            lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    if pilot2:
        # Find best embedding model
        best_emb = None
        best_n_sig = -1
        for key, data in pilot2.items():
            n_sig = data.get("n_significant_features", 0)
            if n_sig > best_n_sig:
                best_n_sig = n_sig
                best_emb = data.get("embedding_model", "?")
        if best_emb:
            lines.append(f"- **Best embedding model**: {best_emb} ({best_n_sig} significant features)")

    if pilot3:
        # Find best marker category
        all_rhos = {}
        for data in pilot3.values():
            for cat, corr in data.get("correlations", {}).items():
                all_rhos.setdefault(cat, []).append(abs(corr["spearman_rho"]))
        if all_rhos:
            best_cat = max(all_rhos, key=lambda k: np.mean(all_rhos[k]))
            lines.append(f"- **Best marker category**: {best_cat} "
                        f"(mean |ρ|={np.mean(all_rhos[best_cat]):.3f})")

    lines.append("")
    return "\n".join(lines)


def main():
    print("="*60)
    print("PILOT SUMMARY REPORT")
    print("="*60)

    pilot1 = load_json("pilot1_results.json")
    pilot2 = load_json("pilot2_results.json")
    pilot3 = load_json("pilot3_results.json")

    if not any([pilot1, pilot2, pilot3]):
        print("No pilot results found. Run the pilots first.")
        return

    # Generate plots
    plot_paths = []
    if pilot2:
        plot_paths = generate_similarity_plots(pilot2)

    # Generate report
    report = generate_report(pilot1, pilot2, pilot3, plot_paths)

    # Save
    report_path = RESULTS_DIR / "pilot_summary.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n✓ Summary report → {report_path}")

    # Print to console
    print("\n" + report)


if __name__ == "__main__":
    main()
