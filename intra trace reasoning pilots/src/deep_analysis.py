"""
Deep Analysis: Epistemic State Transition Analysis

Goes beyond the basic pilots to extract publishable insights and
RL-relevant features. Key analyses:

1. EPISTEMIC TRAJECTORY PROFILING
   - Classify trace regions into epistemic states: Exploring, Committing, Verifying, Backtracking
   - Map the "epistemic trajectory" of each trace
   - Compare trajectories of correct vs incorrect answers

2. TEMPORAL MARKER × TRANSITION COUPLING
   - Do uncertainty markers (Wait, Actually, Let me check) temporally
     precede hypothesis transitions? If so → causal signal
   - Could serve as RL "action" detectors

3. CROSS-MODEL CONSISTENCY
   - Do different models produce similar epistemic patterns for the same question?
   - Questions where ALL models struggle → inherently ambiguous
   - Questions where models diverge → interesting for RL exploration

4. COMMITMENT DYNAMICS DEEP DIVE
   - Plot per-question "commitment curves" (drift-to-final over trace depth)
   - Identify "premature commitment" (early high similarity → wrong)
   - Identify "productive exploration" (late convergence → correct)

Usage:
    python deep_analysis.py  # Run all analyses on existing traces + embeddings
"""

import json
import re
import sys
import gc
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from scipy.stats import spearmanr, mannwhitneyu, pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    REASONING_MODELS,
    EMBEDDING_MODELS,
    TRACES_DIR,
    RESULTS_DIR,
    UNCERTAINTY_MARKERS,
    PILOT_CONFIG,
)
from generate_traces import load_traces

ANALYSIS_DIR = RESULTS_DIR / "deep_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. EPISTEMIC STATE CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────────────

# Epistemic state markers (regex patterns)
EPISTEMIC_MARKERS = {
    "exploring": [
        r"\bLet me think\b", r"\bLet's consider\b", r"\bLooking at\b",
        r"\bFirst\b", r"\bLet me analyze\b", r"\bWhat if\b",
        r"\bpossible\b", r"\bmaybe\b", r"\bcould be\b",
        r"\bI need to consider\b", r"\bLet me start\b",
    ],
    "committing": [
        r"\btherefore\b", r"\bthus\b", r"\bso the answer\b",
        r"\bmost likely\b", r"\bI think it's\b", r"\bthis points to\b",
        r"\bthe diagnosis is\b", r"\bthis suggests\b",
        r"\bconsistent with\b", r"\bstrongly suggests\b",
    ],
    "verifying": [
        r"\bLet me verify\b", r"\bLet me check\b", r"\bdouble-check\b",
        r"\bto confirm\b", r"\bchecking\b", r"\bDoes this make sense\b",
        r"\bLet me make sure\b", r"\bIs this consistent\b",
    ],
    "backtracking": [
        r"\bWait\b", r"\bActually\b", r"\bNo,\b", r"\bHmm\b",
        r"\bthat's wrong\b", r"\bLet me reconsider\b",
        r"\bOn second thought\b", r"\bI was wrong\b",
        r"\bcorrection\b", r"\bhold on\b",
    ],
}


def classify_sentence_state(sentence: str) -> str:
    """Classify a sentence into an epistemic state."""
    scores = {}
    for state, patterns in EPISTEMIC_MARKERS.items():
        score = sum(1 for p in patterns if re.search(p, sentence, re.IGNORECASE))
        scores[state] = score

    if max(scores.values()) == 0:
        return "neutral"
    return max(scores, key=scores.get)


def compute_epistemic_trajectory(thinking_text: str) -> Dict:
    """
    Compute the epistemic trajectory of a reasoning trace.

    Returns:
        Dict with trajectory sequence, state distributions, and transition matrix.
    """
    from pilot1_transitions import split_into_sentences

    sentences = split_into_sentences(thinking_text)
    if len(sentences) < 3:
        return None

    # Classify each sentence
    states = [classify_sentence_state(s) for s in sentences]

    # State distribution
    state_counts = Counter(states)
    total = len(states)
    state_dist = {s: state_counts.get(s, 0) / total for s in
                  ["exploring", "committing", "verifying", "backtracking", "neutral"]}

    # Transition matrix: P(state_j | state_i)
    all_states = ["exploring", "committing", "verifying", "backtracking", "neutral"]
    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(states) - 1):
        transitions[states[i]][states[i+1]] += 1

    # Normalize
    trans_matrix = {}
    for s1 in all_states:
        row_total = sum(transitions[s1].values())
        trans_matrix[s1] = {
            s2: transitions[s1][s2] / max(row_total, 1)
            for s2 in all_states
        }

    # Key metrics
    # 1. Exploration ratio: fraction of trace spent exploring
    explore_ratio = state_dist.get("exploring", 0)
    # 2. Backtrack ratio: how much backtracking happens
    backtrack_ratio = state_dist.get("backtracking", 0)
    # 3. Final state: the last non-neutral state
    final_state = "neutral"
    for s in reversed(states):
        if s != "neutral":
            final_state = s
            break
    # 4. Commitment position: first time "committing" appears (normalized)
    first_commit = 1.0
    for i, s in enumerate(states):
        if s == "committing":
            first_commit = i / len(states)
            break
    # 5. Exploration before commit: amount of exploring before first commit
    explore_before_commit = sum(1 for s in states[:int(first_commit * len(states))]
                                if s == "exploring") / max(1, int(first_commit * len(states)))

    return {
        "states": states,
        "state_distribution": state_dist,
        "transition_matrix": trans_matrix,
        "n_sentences": len(sentences),
        "explore_ratio": explore_ratio,
        "backtrack_ratio": backtrack_ratio,
        "commit_ratio": state_dist.get("committing", 0),
        "verify_ratio": state_dist.get("verifying", 0),
        "final_state": final_state,
        "first_commit_position": first_commit,
        "explore_before_commit": explore_before_commit,
    }


def run_epistemic_analysis(model_names: List[str]) -> Dict:
    """Run epistemic trajectory analysis across all models."""
    results = {}

    for model_name in model_names:
        try:
            traces = load_traces(model_name)
        except FileNotFoundError:
            continue

        print(f"\n{'='*60}")
        print(f"EPISTEMIC ANALYSIS: {model_name}")
        print(f"{'='*60}")

        correct_trajectories = []
        incorrect_trajectories = []

        for trace in traces:
            if not trace.thinking_text or len(trace.thinking_text) < 50:
                continue
            traj = compute_epistemic_trajectory(trace.thinking_text)
            if traj is None:
                continue
            traj["question_id"] = trace.question_id
            traj["is_correct"] = trace.is_correct

            if trace.is_correct:
                correct_trajectories.append(traj)
            else:
                incorrect_trajectories.append(traj)

        if not correct_trajectories or not incorrect_trajectories:
            print(f"  Insufficient data (correct={len(correct_trajectories)}, incorrect={len(incorrect_trajectories)})")
            continue

        # Compare correct vs incorrect
        metrics = [
            "explore_ratio", "backtrack_ratio", "commit_ratio",
            "verify_ratio", "first_commit_position", "explore_before_commit"
        ]

        stats = {}
        for metric in metrics:
            c_vals = np.array([t[metric] for t in correct_trajectories])
            w_vals = np.array([t[metric] for t in incorrect_trajectories])

            if len(c_vals) < 2 or len(w_vals) < 2:
                continue

            try:
                _, p = mannwhitneyu(c_vals, w_vals, alternative="two-sided")
            except ValueError:
                p = 1.0

            # Cohen's d
            pooled_std = np.sqrt(((len(c_vals)-1)*np.var(c_vals, ddof=1) +
                                  (len(w_vals)-1)*np.var(w_vals, ddof=1)) /
                                 (len(c_vals) + len(w_vals) - 2))
            d = (np.mean(c_vals) - np.mean(w_vals)) / max(pooled_std, 1e-9)

            stats[metric] = {
                "correct_mean": float(np.mean(c_vals)),
                "incorrect_mean": float(np.mean(w_vals)),
                "p_value": float(p),
                "cohens_d": float(d),
                "significant": p < 0.05,
            }

            sig = "✓" if p < 0.05 else "✗"
            print(f"  [{sig}] {metric:25s}: "
                  f"correct={np.mean(c_vals):.3f}, "
                  f"incorrect={np.mean(w_vals):.3f}, "
                  f"p={p:.4f}, d={d:.3f}")

        # Final state analysis
        correct_finals = Counter(t["final_state"] for t in correct_trajectories)
        incorrect_finals = Counter(t["final_state"] for t in incorrect_trajectories)
        print(f"\n  Final epistemic state (correct):   {dict(correct_finals)}")
        print(f"  Final epistemic state (incorrect): {dict(incorrect_finals)}")

        results[model_name] = {
            "n_correct": len(correct_trajectories),
            "n_incorrect": len(incorrect_trajectories),
            "stats": stats,
            "correct_finals": dict(correct_finals),
            "incorrect_finals": dict(incorrect_finals),
            "correct_trajectories": correct_trajectories,
            "incorrect_trajectories": incorrect_trajectories,
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 2. TEMPORAL MARKER × TRANSITION COUPLING
# ──────────────────────────────────────────────────────────────────────────────

def run_temporal_coupling(model_names: List[str]) -> Dict:
    """
    Analyze whether uncertainty markers temporally precede hypothesis transitions.

    If "Wait" / "Actually" appears 1-3 sentences before a similarity drop,
    this suggests the model "knows" it's about to shift hypotheses —
    a causal signal useful for RL reward shaping.
    """
    from pilot1_transitions import split_into_sentences

    results = {}

    for model_name in model_names:
        try:
            traces = load_traces(model_name)
        except FileNotFoundError:
            continue

        print(f"\n{'='*60}")
        print(f"TEMPORAL COUPLING: {model_name}")
        print(f"{'='*60}")

        all_marker_positions = []  # normalized positions of markers
        all_transition_positions = []  # will be filled from pilot1 results

        # Load pilot1 results to get transition positions
        pilot1_file = RESULTS_DIR / "pilot1_results.json"
        pilot1_data = {}
        if pilot1_file.exists():
            with open(pilot1_file) as f:
                pilot1_data = json.load(f)

        # For each trace, find marker positions and check proximity to transitions
        coupling_events = []  # (marker_type, distance_to_nearest_transition)

        for trace in traces:
            if not trace.thinking_text or len(trace.thinking_text) < 50:
                continue

            sentences = split_into_sentences(trace.thinking_text)
            if len(sentences) < 5:
                continue

            # Find marker positions (sentence-level)
            marker_positions = {}
            for i, sent in enumerate(sentences):
                for cat, patterns in UNCERTAINTY_MARKERS.items():
                    if any(re.search(p, sent, re.IGNORECASE) for p in patterns):
                        marker_positions.setdefault(cat, []).append(i)

            # Find transition positions from pilot1 results
            key = f"{model_name}__bge-m3"
            transition_indices = []
            if key in pilot1_data:
                for td in pilot1_data[key].get("trace_details", []):
                    if td["question_id"] == trace.question_id:
                        transition_indices = td.get("transition_indices", [])
                        break

            if not transition_indices:
                continue

            # Compute temporal coupling: for each marker, find nearest transition
            for cat, positions in marker_positions.items():
                for pos in positions:
                    # Find nearest SUBSEQUENT transition (markers that predict transitions)
                    future_trans = [t for t in transition_indices if t > pos]
                    if future_trans:
                        distance = min(future_trans) - pos
                        coupling_events.append({
                            "model": model_name,
                            "question_id": trace.question_id,
                            "is_correct": trace.is_correct,
                            "marker_type": cat,
                            "marker_position": pos / len(sentences),
                            "distance_to_transition": distance,
                            "distance_normalized": distance / len(sentences),
                        })

        if not coupling_events:
            print(f"  No coupling events found")
            continue

        # Analyze coupling distance by marker type
        by_category = defaultdict(list)
        for ev in coupling_events:
            by_category[ev["marker_type"]].append(ev["distance_to_transition"])

        print(f"  Total coupling events: {len(coupling_events)}")
        for cat, distances in by_category.items():
            within_3 = sum(1 for d in distances if d <= 3)
            pct = 100 * within_3 / len(distances) if distances else 0
            print(f"  {cat:10s}: mean distance={np.mean(distances):.1f} sentences, "
                  f"{within_3}/{len(distances)} ({pct:.0f}%) within 3 sentences of transition")

        # Key question: Do "revise" markers (Wait, Actually) predict transitions
        # more than random? Expected by chance: ~3/N where N=trace_length
        revise_dists = by_category.get("revise", [])
        if revise_dists:
            mean_trace_len = np.mean([len(split_into_sentences(t.thinking_text))
                                      for t in traces if t.thinking_text and len(t.thinking_text) > 50])
            expected_by_chance = 3 / mean_trace_len * 100
            actual = 100 * sum(1 for d in revise_dists if d <= 3) / len(revise_dists)
            print(f"\n  *** KEY FINDING: 'revise' markers within 3 steps of transition: {actual:.1f}%")
            print(f"  *** Expected by random chance: {expected_by_chance:.1f}%")
            print(f"  *** This is {actual/max(expected_by_chance, 0.1):.1f}x above chance")
            if actual > expected_by_chance * 2:
                print(f"  *** → STRONG temporal coupling! Markers PREDICT transitions.")
                print(f"  *** RL implication: 'revise' markers can serve as reward signals.")

        results[model_name] = {
            "n_coupling_events": len(coupling_events),
            "by_category": {cat: {
                "count": len(dists),
                "mean_distance": float(np.mean(dists)),
                "within_3_sentences": sum(1 for d in dists if d <= 3),
                "pct_within_3": float(100 * sum(1 for d in dists if d <= 3) / max(len(dists), 1)),
            } for cat, dists in by_category.items()},
            "coupling_events": coupling_events,
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 3. CROSS-MODEL CONSISTENCY
# ──────────────────────────────────────────────────────────────────────────────

def run_cross_model_analysis(model_names: List[str]) -> Dict:
    """
    Analyze cross-model consistency on the same questions.

    Questions where ALL models get correct → "easy" (well-explored)
    Questions where ALL models fail → "hard" (inherently ambiguous)
    Questions with mixed results → interesting for RL (exploration needed)
    """
    print(f"\n{'='*60}")
    print("CROSS-MODEL CONSISTENCY ANALYSIS")
    print(f"{'='*60}")

    # Load all traces across models
    all_traces = {}
    for model_name in model_names:
        try:
            traces = load_traces(model_name)
            all_traces[model_name] = {t.question_id: t for t in traces}
        except FileNotFoundError:
            continue

    if len(all_traces) < 2:
        print("  Need at least 2 models")
        return {}

    # Find common questions
    common_ids = set.intersection(*[set(traces.keys()) for traces in all_traces.values()])
    print(f"  Models: {list(all_traces.keys())}")
    print(f"  Common questions: {len(common_ids)}")

    # Categorize questions
    all_correct = []
    all_wrong = []
    mixed = []
    question_profiles = {}

    for qid in sorted(common_ids):
        correct_models = [m for m in all_traces if all_traces[m][qid].is_correct]
        wrong_models = [m for m in all_traces if not all_traces[m][qid].is_correct]

        profile = {
            "question_id": qid,
            "n_correct": len(correct_models),
            "n_wrong": len(wrong_models),
            "correct_models": correct_models,
            "wrong_models": wrong_models,
            "difficulty": "easy" if len(wrong_models) == 0 else
                          "hard" if len(correct_models) == 0 else "mixed",
        }
        question_profiles[qid] = profile

        if len(wrong_models) == 0:
            all_correct.append(qid)
        elif len(correct_models) == 0:
            all_wrong.append(qid)
        else:
            mixed.append(qid)

    print(f"\n  All correct (easy): {len(all_correct)} questions")
    print(f"  All wrong (hard):   {len(all_wrong)} questions")
    print(f"  Mixed:              {len(mixed)} questions")

    # For mixed questions: analyze what differences exist in traces
    if mixed:
        print(f"\n  Mixed question details:")
        for qid in mixed[:5]:
            prof = question_profiles[qid]
            print(f"    {qid}: correct by {prof['correct_models']}, wrong by {prof['wrong_models']}")

    # Agreement matrix
    models = list(all_traces.keys())
    agreement = np.zeros((len(models), len(models)))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            n_agree = sum(1 for qid in common_ids
                         if all_traces[m1][qid].is_correct == all_traces[m2][qid].is_correct)
            agreement[i][j] = n_agree / max(len(common_ids), 1)

    print(f"\n  Agreement matrix (% same answer correctness):")
    header = "            " + "  ".join(f"{m:>15s}" for m in models)
    print(header)
    for i, m in enumerate(models):
        row = f"  {m:>10s}: " + "  ".join(f"{agreement[i][j]:>14.1%}" for j in range(len(models)))
        print(row)

    # Key insight: Mixed questions are the most interesting for RL —
    # they represent "epistemic uncertainty" where the model's reasoning path matters
    results = {
        "n_common_questions": len(common_ids),
        "n_easy": len(all_correct),
        "n_hard": len(all_wrong),
        "n_mixed": len(mixed),
        "easy_questions": all_correct,
        "hard_questions": all_wrong,
        "mixed_questions": mixed,
        "question_profiles": question_profiles,
        "agreement_matrix": agreement.tolist(),
        "model_names": models,
    }

    # RL-relevant insight
    print(f"\n  *** RL INSIGHT: {len(mixed)} 'mixed' questions = epistemic uncertainty frontier")
    print(f"  *** These are where reasoning PATH matters (same question, different outcomes)")
    print(f"  *** → Ideal candidates for RL exploration/exploitation training")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 4. COMMITMENT DYNAMICS DEEP DIVE
# ──────────────────────────────────────────────────────────────────────────────

def run_commitment_deep_dive(model_names: List[str]) -> Dict:
    """
    Deep analysis of commitment dynamics: when and how models commit to answers.

    Key hypothesis: "Premature commitment" = high early similarity → wrong answer
    "Productive exploration" = late convergence → correct answer
    """
    print(f"\n{'='*60}")
    print("COMMITMENT DYNAMICS DEEP DIVE")
    print(f"{'='*60}")

    # Load pilot2 results for similarity curves
    pilot2_file = RESULTS_DIR / "pilot2_results.json"
    if not pilot2_file.exists():
        print("  Need pilot2_results.json first")
        return {}

    with open(pilot2_file) as f:
        pilot2_data = json.load(f)

    results = {}

    for key, data in pilot2_data.items():
        if "all_features" not in data:
            continue

        model_name = data.get("reasoning_model", "?")
        emb_name = data.get("embedding_model", "?")

        print(f"\n  {model_name} + {emb_name}")

        correct = [f for f in data["all_features"] if f["is_correct"]]
        incorrect = [f for f in data["all_features"] if not f["is_correct"]]

        if not correct or not incorrect:
            continue

        # Analyze commitment timing
        # For each trace, compute: at what depth (%) does similarity first exceed 0.7?
        def commitment_depth(sims, threshold=0.7):
            for i, s in enumerate(sims):
                if s >= threshold:
                    return i / len(sims)
            return 1.0  # never committed

        correct_commitment = [commitment_depth(f.get("similarities", [])) for f in correct]
        incorrect_commitment = [commitment_depth(f.get("similarities", [])) for f in incorrect]

        if correct_commitment and incorrect_commitment:
            c_mean = np.mean(correct_commitment)
            w_mean = np.mean(incorrect_commitment)
            print(f"    Commitment depth (τ=0.7): correct={c_mean:.3f}, incorrect={w_mean:.3f}")
            if w_mean < c_mean:
                print(f"    *** PREMATURE COMMITMENT: incorrect answers commit {(c_mean-w_mean)*100:.1f}% earlier!")
                print(f"    *** RL signal: early commitment → penalize, late exploration → reward")

        # Analyze "exploration breadth" — how much similarity variance in first half
        def first_half_variance(sims):
            half = len(sims) // 2
            if half < 2:
                return 0
            return float(np.var(sims[:half]))

        correct_var = [first_half_variance(f.get("similarities", [])) for f in correct]
        incorrect_var = [first_half_variance(f.get("similarities", [])) for f in incorrect]

        if correct_var and incorrect_var:
            c_var = np.mean(correct_var)
            w_var = np.mean(incorrect_var)
            print(f"    First-half similarity variance: correct={c_var:.4f}, incorrect={w_var:.4f}")
            if w_var > c_var:
                print(f"    *** Incorrect traces show MORE first-half turbulence → unstable exploration")

        # Convergence smoothness: does the trace smoothly converge or oscillate?
        def convergence_smoothness(sims):
            if len(sims) < 4:
                return 0
            # Second derivative of similarity curve
            diffs = np.diff(sims)
            second_diffs = np.diff(diffs)
            return float(np.mean(np.abs(second_diffs)))

        correct_smooth = [convergence_smoothness(f.get("similarities", [])) for f in correct]
        incorrect_smooth = [convergence_smoothness(f.get("similarities", [])) for f in incorrect]

        if correct_smooth and incorrect_smooth:
            c_s = np.mean(correct_smooth)
            w_s = np.mean(incorrect_smooth)
            print(f"    Convergence roughness: correct={c_s:.4f}, incorrect={w_s:.4f}")
            if w_s > c_s:
                print(f"    *** Incorrect traces have ROUGHER convergence → oscillating reasoning")

        results[key] = {
            "model": model_name,
            "embedding": emb_name,
            "correct_commitment_mean": float(np.mean(correct_commitment)) if correct_commitment else 0,
            "incorrect_commitment_mean": float(np.mean(incorrect_commitment)) if incorrect_commitment else 0,
            "correct_first_half_var": float(np.mean(correct_var)) if correct_var else 0,
            "incorrect_first_half_var": float(np.mean(incorrect_var)) if incorrect_var else 0,
            "correct_convergence_roughness": float(np.mean(correct_smooth)) if correct_smooth else 0,
            "incorrect_convergence_roughness": float(np.mean(incorrect_smooth)) if incorrect_smooth else 0,
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def save_json(data, filename):
    path = ANALYSIS_DIR / filename
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    clean = json.loads(json.dumps(data, default=convert))
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"✓ Saved → {path}")


def main():
    models = [m for m in REASONING_MODELS.keys()
              if (TRACES_DIR / m / "traces.jsonl").exists()]

    print("="*60)
    print("DEEP ANALYSIS: Epistemic State Transitions")
    print("="*60)
    print(f"Models with traces: {models}")

    # 1. Epistemic trajectory analysis
    epistemic_results = run_epistemic_analysis(models)
    save_json(epistemic_results, "epistemic_trajectories.json")

    # 2. Temporal coupling
    coupling_results = run_temporal_coupling(models)
    save_json(coupling_results, "temporal_coupling.json")

    # 3. Cross-model consistency
    cross_model_results = run_cross_model_analysis(models)
    save_json(cross_model_results, "cross_model_consistency.json")

    # 4. Commitment dynamics deep dive
    commitment_results = run_commitment_deep_dive(models)
    save_json(commitment_results, "commitment_dynamics.json")

    # Summary
    print(f"\n{'='*60}")
    print("DEEP ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"  All results saved to: {ANALYSIS_DIR}")
    print(f"  Files: epistemic_trajectories.json, temporal_coupling.json,")
    print(f"         cross_model_consistency.json, commitment_dynamics.json")


if __name__ == "__main__":
    main()
