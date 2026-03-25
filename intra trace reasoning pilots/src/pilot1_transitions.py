"""
Pilot 1: Can embedding models detect hypothesis transitions?

Embeds each sentence of a reasoning trace, computes consecutive
cosine similarity, identifies drops that correspond to hypothesis
transitions (e.g., pneumonia → TB → lymphoma).

Improvements over initial version:
- Z-score based transition detection (more robust than fixed-neighbor comparison)
- Proper Qwen3-Embedding-8B backend with task instruction prefix
- Better sentence segmentation handling markdown/Chinese punctuation

Usage:
    python pilot1_transitions.py                        # All models, both embeddings
    python pilot1_transitions.py --model qwq-32b        # Single reasoning model
    python pilot1_transitions.py --embedding bge-m3     # Single embedding model
"""

import argparse
import json
import re
import sys
import gc
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    REASONING_MODELS,
    EMBEDDING_MODELS,
    TRACES_DIR,
    EMBEDDINGS_DIR,
    RESULTS_DIR,
    PILOT_CONFIG,
    EmbeddingModelConfig,
)
from generate_traces import load_traces, ReasoningTrace


# ─── Sentence Segmentation ───────────────────────────────────────────────────

ABBREVIATIONS = {"Dr.", "Mr.", "Mrs.", "Ms.", "e.g.", "i.e.", "vs.", "etc.",
                 "approx.", "Fig.", "Eq.", "No.", "Vol.", "Inc.", "Ltd.",
                 "St.", "Jr.", "Sr.", "Prof.", "Dept."}


def split_into_sentences(text: str) -> List[str]:
    """
    Split thinking text into sentences using linguistic boundaries.

    Handles:
    - Abbreviations and decimal numbers
    - Paragraph breaks
    - Markdown formatting (**bold**, - list items, numbered lists)
    - Chinese punctuation (。！？) from Qwen models
    """
    if not text or len(text.strip()) < 10:
        return []

    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)

    # Strip markdown bold/italic markers
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)

    # Split on paragraph breaks first
    paragraphs = re.split(r"\n\s*\n", text)

    sentences = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Protect abbreviations and decimals
        protected = para
        for abbr in ABBREVIATIONS:
            protected = protected.replace(abbr, abbr.replace(".", "§DOT§"))
        protected = re.sub(r"(\d)\.(\d)", r"\1§DOT§\2", protected)

        # Split on sentence boundaries (English + Chinese)
        parts = re.split(
            r'(?<=[.!?。！？])\s+(?=[A-Z\u4e00-\u9fff])|(?<=[.!?。！？])\s*\n',
            protected
        )

        for part in parts:
            part = part.replace("§DOT§", ".").strip()
            if part and len(part.split()) >= PILOT_CONFIG.min_sentence_words:
                sentences.append(part)

    # Also split on numbered/bulleted list items within sentences
    final = []
    for sent in sentences:
        sub = re.split(r"\n\s*(?=\d+[\.)\)]\s|\-\s|\*\s|•\s)", sent)
        for s in sub:
            s = s.strip()
            if s and len(s.split()) >= PILOT_CONFIG.min_sentence_words:
                final.append(s)

    return final


# ─── Embedding ────────────────────────────────────────────────────────────────


class EmbeddingBackend:
    """
    Unified embedding backend supporting both BGE-M3 (via sentence-transformers)
    and Qwen3-Embedding-8B (via transformers with task instruction).
    """

    # Qwen3-Embedding-8B requires a task instruction prefix for proper embeddings
    QWEN_TASK_INSTRUCTION = "Instruct: Retrieve semantically similar passages.\nQuery: "

    def __init__(self, hf_id: str, max_seq_length: int):
        self.hf_id = hf_id
        self.max_seq_length = max_seq_length

        self.is_qwen = "qwen3" in hf_id.lower()

        if not self.is_qwen:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                hf_id,
                device="cuda" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
            )
            self.model.max_seq_length = max_seq_length

        else:
            from transformers import AutoTokenizer, AutoModel
            self.tok = AutoTokenizer.from_pretrained(
                hf_id,
                trust_remote_code=True,
                padding_side="right",
            )
            self.qwen_model = AutoModel.from_pretrained(
                hf_id,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            ).eval()

    def encode(self, sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.array([])

        if not self.is_qwen:
            embs = self.model.encode(
                sentences,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.array(embs)

        # QWEN3-EMBEDDING-8B PATH
        # The model requires task-prefixed inputs for proper semantic encoding
        prefixed = [self.QWEN_TASK_INSTRUCTION + s for s in sentences]

        all_embs = []
        batch_size = 8  # Smaller batches — 8B model uses more VRAM per batch
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i:i + batch_size]
            inputs = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.qwen_model.device)

            with torch.no_grad():
                out = self.qwen_model(**inputs)

            # Mean pooling with attention mask
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

            # L2 normalize
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.float().cpu().numpy())

        return np.concatenate(all_embs, axis=0)

    def cleanup(self):
        """Free GPU memory."""
        if self.is_qwen:
            del self.qwen_model
            del self.tok
        else:
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_embedding_model(config: EmbeddingModelConfig):
    return EmbeddingBackend(
        config.hf_id,
        config.max_seq_length
    )


def encode_sentences(model, sentences: List[str]) -> np.ndarray:
    return model.encode(sentences)


# ─── Similarity Analysis ─────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def compute_similarity_curve(embeddings: np.ndarray) -> List[float]:
    """Cosine similarity between consecutive sentence embeddings."""
    return [cosine_sim(embeddings[i], embeddings[i+1])
            for i in range(len(embeddings) - 1)]


def detect_transitions(similarities: List[float],
                       drop_threshold: float = 0.15) -> List[int]:
    """
    Detect hypothesis transitions using a z-score approach combined
    with an absolute threshold.

    A transition is detected when a similarity value is:
    - More than 1.5 standard deviations below the mean, OR
    - Drops more than `drop_threshold` below local context (±2 neighbors)

    This is more robust than the original ±1 neighbor approach, handling
    traces with varying baseline similarity levels.
    """
    if len(similarities) < 3:
        return []

    transitions = []
    sims = np.array(similarities)
    mean_sim = np.mean(sims)
    std_sim = np.std(sims)

    for i in range(len(sims)):
        is_transition = False

        # Method 1: Z-score — similarity is unusually low
        if std_sim > 0.01:  # avoid div by zero on flat curves
            z_score = (sims[i] - mean_sim) / std_sim
            if z_score < -1.5:
                is_transition = True

        # Method 2: Local context drop (wider window of ±2 neighbors)
        neighbors = []
        for j in range(max(0, i - 2), min(len(sims), i + 3)):
            if j != i:
                neighbors.append(sims[j])
        if neighbors:
            local_mean = np.mean(neighbors)
            drop = local_mean - sims[i]
            if drop > drop_threshold:
                is_transition = True

        if is_transition:
            transitions.append(i)

    return transitions


# ─── Main Pilot 1 Analysis ───────────────────────────────────────────────────

def run_pilot1(
    model_names: List[str],
    embedding_names: List[str],
) -> Dict:
    """
    Run Pilot 1: Hypothesis Transition Detection.

    For each (reasoning_model, embedding_model) pair:
    1. Load traces
    2. Segment into sentences
    3. Embed with the embedding model
    4. Compute cosine similarity curves
    5. Detect transitions
    6. Print analysis
    """
    results = {}

    for emb_name in embedding_names:
        emb_config = EMBEDDING_MODELS[emb_name]
        print(f"\n{'='*60}")
        print(f"EMBEDDING MODEL: {emb_name} ({emb_config.hf_id})")
        print(f"{'='*60}")

        emb_model = load_embedding_model(emb_config)

        for model_name in model_names:
            try:
                traces = load_traces(model_name)
            except FileNotFoundError:
                print(f"  ⚠ No traces for {model_name}, skipping")
                continue

            key = f"{model_name}__{emb_name}"
            print(f"\n  Analyzing: {model_name}")

            # Process each trace
            trace_results = []
            for trace in traces:
                if not trace.thinking_text or len(trace.thinking_text) < 50:
                    continue

                # Segment
                sentences = split_into_sentences(trace.thinking_text)
                if len(sentences) < 3:
                    continue

                # Embed
                embeddings = encode_sentences(emb_model, sentences)
                if len(embeddings) < 3:
                    continue

                # Similarity curve
                similarities = compute_similarity_curve(embeddings)

                # Detect transitions
                transitions = detect_transitions(
                    similarities, PILOT_CONFIG.cosine_drop_threshold
                )

                # Compute slope (convergence metric)
                x = np.arange(len(similarities))
                slope = float(np.polyfit(x, similarities, 1)[0]) if len(similarities) >= 2 else 0.0

                trace_results.append({
                    "question_id": trace.question_id,
                    "is_correct": trace.is_correct,
                    "n_sentences": len(sentences),
                    "similarities": similarities,
                    "transition_indices": transitions,
                    "n_transitions": len(transitions),
                    "mean_similarity": float(np.mean(similarities)),
                    "min_similarity": float(np.min(similarities)),
                    "max_similarity": float(np.max(similarities)),
                    "std_similarity": float(np.std(similarities)),
                    "similarity_slope": slope,
                    # Store sentences at transition points for inspection
                    "transition_contexts": [
                        {
                            "before": sentences[t] if t < len(sentences) else "",
                            "after": sentences[t+1] if t+1 < len(sentences) else "",
                            "similarity": similarities[t],
                        }
                        for t in transitions
                    ],
                })

            if not trace_results:
                print(f"    No valid traces to analyze")
                continue

            # Aggregate stats
            n_trans = [r["n_transitions"] for r in trace_results]
            has_trans = sum(1 for r in trace_results if r["n_transitions"] > 0)

            results[key] = {
                "reasoning_model": model_name,
                "embedding_model": emb_name,
                "n_traces_analyzed": len(trace_results),
                "pct_with_transitions": 100 * has_trans / len(trace_results),
                "mean_transitions_per_trace": float(np.mean(n_trans)),
                "median_transitions": float(np.median(n_trans)),
                "mean_sentences_per_trace": float(np.mean([r["n_sentences"] for r in trace_results])),
                "mean_similarity": float(np.mean([r["mean_similarity"] for r in trace_results])),
                "trace_details": trace_results,
            }

            # Print summary
            print(f"    Traces analyzed: {len(trace_results)}")
            print(f"    With transitions: {has_trans} ({results[key]['pct_with_transitions']:.0f}%)")
            print(f"    Mean transitions/trace: {results[key]['mean_transitions_per_trace']:.1f}")
            print(f"    Mean sentences/trace: {results[key]['mean_sentences_per_trace']:.1f}")
            print(f"    Mean cosine similarity: {results[key]['mean_similarity']:.3f}")

            # Show transition examples
            for r in trace_results[:3]:
                if r["transition_contexts"]:
                    print(f"\n    Example transitions ({r['question_id']}), correct={r['is_correct']}:")
                    for tc in r["transition_contexts"][:2]:
                        print(f"      sim={tc['similarity']:.3f}")
                        print(f"      BEFORE: {tc['before'][:100]}...")
                        print(f"      AFTER:  {tc['after'][:100]}...")

        # Cleanup embedding model
        emb_model.cleanup()

    return results


def save_results(results: Dict, filename: str = "pilot1_results.json"):
    """Save results to JSON, merging with existing file."""
    output = RESULTS_DIR / filename
    # Convert numpy types for JSON serialization
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
    print(f"\n✓ Results saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Pilot 1: Hypothesis Transition Detection")
    parser.add_argument("--model", type=str, default=None,
                       choices=list(REASONING_MODELS.keys()))
    parser.add_argument("--embedding", type=str, default=None,
                       choices=list(EMBEDDING_MODELS.keys()))
    args = parser.parse_args()

    models = [args.model] if args.model else list(REASONING_MODELS.keys())
    embeddings = [args.embedding] if args.embedding else list(EMBEDDING_MODELS.keys())

    print("="*60)
    print("PILOT 1: Hypothesis Transition Detection")
    print("="*60)
    print(f"Reasoning models: {models}")
    print(f"Embedding models: {embeddings}")

    results = run_pilot1(models, embeddings)
    save_results(results)

    # Summary
    print("\n" + "="*60)
    print("PILOT 1 SUMMARY")
    print("="*60)
    for key, data in results.items():
        print(f"  {data['reasoning_model']} + {data['embedding_model']}:")
        print(f"    {data['pct_with_transitions']:.0f}% traces have transitions, "
              f"mean={data['mean_transitions_per_trace']:.1f}/trace")


if __name__ == "__main__":
    main()
