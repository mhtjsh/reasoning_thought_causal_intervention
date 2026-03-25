# Pilot Validation Plan (Corrected)

## Prompt Decision

**Choice: Minimal prompt.** Here's why, from an A* paper perspective:

| A* Conference Requirement | Why Minimal Prompt Satisfies It |
|---|---|
| **Generalizability** | Same prompt structure works for MedQA AND MATH — method contribution, not prompt engineering |
| **Reproducibility** | No complex chain-of-prompt dependencies a reviewer needs to replicate |
| **Clean ablation story** | If results come from a simple prompt, reviewers attribute the gain to *your method*, not to prompt tricks |
| **Reviewer concern averted** | Over-engineered prompts invite: "This works because of the prompt, not the method" |

The `<think>` trace emerges from the **model architecture** (QwQ, Qwen3 thinking mode), not from us telling the model how to reason. That's the whole point — we're measuring the model's *natural* reasoning dynamics.

### Prompt (Medical — MedQA)
```
You are a medical expert. Based on the following clinical case,
what is the most likely diagnosis?

{clinical_vignette}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}
```

### Prompt (Math — MATH Level 3-5, for main experiments)
```
Solve the following problem.

{problem_statement}
```

No "think step by step." No physician framing. The model's thinking mode handles that. Clean.

---

## Models

| Model | HuggingFace ID | BF16 VRAM | GPU Plan |
|-------|---------------|-----------|----------|
| QwQ-32B | `Qwen/QwQ-32B` | ~64GB | GPU 0+1 (TP=2) |
| Qwen3.5-27B | `Qwen/Qwen3.5-27B` | ~54GB | GPU 0+1 (TP=2) |
| Qwen3-8B | `Qwen/Qwen3-8B` | ~16GB | GPU 0 |
| Qwen3-4B-Thinking | `Qwen/Qwen3-4B-Thinking-2507` | ~8GB | GPU 0 |

**Embedding models** (compared head-to-head):

| Model | HuggingFace ID | VRAM | Why |
|-------|---------------|------|-----|
| BGE-M3 | `BAAI/bge-m3` | ~1.2GB | Established, efficient |
| Qwen3-Embedding-8B | `Qwen/Qwen3-Embedding-8B` | ~16GB | #1 MTEB, potentially better on reasoning text |

**Fallback**: If neither separates hypotheses → extract model's own hidden states (KV-cache last-token representations) as embeddings, using attention entropy to find hypothesis transition boundaries.

---

## What the Pilots Actually Test

```mermaid
graph TD
    subgraph "Pilot 1: Can Embeddings Detect Hypothesis Shifts?"
        P1["Embed sentence-by-sentence<br/>→ do cosine drops align<br/>with hypothesis transitions?"]
    end

    subgraph "Pilot 2: Do Dynamics Predict Correctness?"
        P2["Given transitions are detectable,<br/>do correct vs incorrect traces<br/>show different convergence patterns?"]
    end

    subgraph "Pilot 3: Are Uncertainty Markers Informative?"
        P3["Separately: do 'Wait', 'probably',<br/>'I'm not sure' correlate<br/>with actual errors?"]
    end

    P1 -->|"Pass"| P2
    P1 -->|"Fail"| KV["Fallback: KV-cache<br/>hidden states + attention<br/>entropy for transitions"]
    KV --> P2

    P2 -->|"Pass"| P3
    P2 -->|"Fail"| STOP["❌ Fundamental signal<br/>issue — rethink"]

    P3 -->|"Pass"| GO["✅ All pass → Main exps"]
    P3 -->|"Fail"| PARTIAL["⚠️ Drop markers,<br/>use embedding dynamics alone"]

    style GO fill:#27ae60,color:#fff
    style PARTIAL fill:#f39c12,color:#fff
    style STOP fill:#c0392b,color:#fff
```

---

## Pilot 1: Hypothesis Transition Detection

**Core question**: Can our embedding models tell when the model switches from reasoning about one diagnosis to another?

### How it works

```mermaid
flowchart TD
    TRACE["Raw <think> trace"] --> SENT["Split into sentences<br/>(period + newline segmentation)"]

    SENT --> S_LIST["s₁: 'The patient presents with fever and dry cough.'<br/>s₂: 'This could indicate pneumonia.'<br/>s₃: 'However, the weight loss and night sweats...'<br/>s₄: 'These constitutional symptoms suggest TB.'<br/>s₅: 'In TB, we typically see...'<br/>s₆: 'But we should also consider lymphoma...'<br/>s₇: 'Given the symptom constellation, TB is most likely.'"]

    S_LIST --> EMB["Embed each sentence<br/>(BGE-M3 AND Qwen3-Emb-8B)"]

    EMB --> SIM["Compute consecutive cosine similarity:<br/>sim(s₁,s₂), sim(s₂,s₃), ..., sim(s₆,s₇)"]

    SIM --> DROPS["Identify similarity drops:<br/>sim(s₂,s₃) = 0.42 ← DROP<br/>sim(s₅,s₆) = 0.38 ← DROP"]

    DROPS --> MANUAL["Manual validation:<br/>Do drops align with actual<br/>hypothesis transitions?<br/><br/>s₂→s₃: pneumonia → TB ✅<br/>s₅→s₆: TB → lymphoma ✅"]

    MANUAL --> METRICS["Metrics:<br/>• Precision: what % of detected drops<br/>  are real hypothesis transitions?<br/>• Recall: what % of real transitions<br/>  produce detectable drops?<br/>• Which embedding model is better?"]

    style DROPS fill:#e74c3c,color:#fff
    style METRICS fill:#2c3e50,color:#fff
```

### Segmentation: Sentence-Level, Not Marker-Level

The trace gets split at sentence boundaries (periods, newlines, paragraph breaks) — these are **linguistic units**, not reasoning signals. Every sentence is a data point for embedding.

| ✅ Sentence-level segmentation | ❌ Marker-level ("Wait", "Actually") |
|---|---|
| Based on language structure | Based on uncertainty signals |
| Every sentence is a data point | Only segments where model hedges |
| Tests if embeddings detect content shifts | Conflates segmentation with Pilot 3 |
| Works the same across all models | Marker vocabulary varies per model |

### What constitutes a "hypothesis transition"

For medical reasoning, a hypothesis transition is when the model shifts its reasoning focus to a **different diagnosis candidate**. Examples:

| Transition? | Example |
|---|---|
| ✅ Yes | "This looks like pneumonia" → "But the weight loss suggests TB" |
| ✅ Yes | "Considering infection" → "Could also be autoimmune" |
| ❌ No | "Pneumonia causes cough" → "Pneumonia also causes fever" (same hypothesis, adding evidence) |
| ❌ No | "The patient is 45" → "with chronic cough" (evidence gathering, no hypothesis yet) |

### Manual annotation protocol (10 traces)

Two annotators independently mark:
1. Which sentences introduce a **new hypothesis**
2. Which sentences provide **evidence for current hypothesis**
3. Which sentences **eliminate a hypothesis**

Compare annotated transitions against embedding similarity drops. Compute precision/recall. Cohen's κ > 0.6 for inter-annotator agreement.

### Pass/fail criteria

| Condition | Result |
|---|---|
| **Precision ≥ 0.6 AND Recall ≥ 0.5** for at least one embedding model | ✅ Pass — embeddings detect hypothesis transitions |
| One model passes, one doesn't | ✅ Pass — use the better model |
| **Both models below threshold** | ❌ Fail — fall back to KV-cache hidden states |

### KV-Cache Fallback (if embeddings fail)

```mermaid
flowchart TD
    FAIL["Embedding models can't<br/>detect hypothesis transitions"] --> KV["Extract hidden states<br/>from reasoning model"]

    KV --> METHOD["For each token position:<br/>1. Extract last-layer hidden state<br/>2. Compute attention entropy<br/>   across all heads"]

    METHOD --> LOW_ATT["Attention entropy peaks =<br/>model redistributing attention<br/>= potential hypothesis transition"]

    LOW_ATT --> PROBE["Validate: do attention<br/>entropy peaks align with<br/>manual hypothesis annotations?"]

    style FAIL fill:#c0392b,color:#fff
    style LOW_ATT fill:#9b59b6,color:#fff
```

This requires the reasoning model to still be loaded (generation + probing in single pass). Only triggered if Pilot 1 fails with both embedding models.

---

## Pilot 2: Embedding Dynamics Predict Correctness

**Core question**: Given that we can detect hypothesis transitions, do the *patterns* of transitions differ between correct and incorrect traces?

```mermaid
flowchart TD
    TRACES["All traces, labeled<br/>correct/incorrect vs MedQA GT"] --> GROUP["Split by correctness"]

    GROUP --> CORRECT["Correct traces:<br/>embedding similarity curves"]
    GROUP --> INCORRECT["Incorrect traces:<br/>embedding similarity curves"]

    CORRECT --> EXPECT_C["Expected pattern:<br/>• Explores multiple hypotheses early<br/>  (low similarity, many drops)<br/>• Converges toward final answer<br/>  (similarity rises at end)<br/>• More transitions = more exploration"]

    INCORRECT --> EXPECT_I["Expected pattern:<br/>• Commits early to one hypothesis<br/>  (high similarity, few drops)<br/>• OR oscillates without converging<br/>  (unstable similarity)"]

    EXPECT_C & EXPECT_I --> STATS["Statistical tests:<br/>1. Mann-Whitney on # of transitions<br/>   (correct vs incorrect)<br/>2. Mean similarity (correct vs incorrect)<br/>3. Convergence slope comparison<br/>4. Effect size (Cohen's d)"]

    STATS --> PASS{"p < 0.05 on<br/>at least 1 metric?"}
    PASS -->|"Yes"| WIN["✅ Embedding dynamics<br/>carry correctness signal"]
    PASS -->|"No"| FAIL2["❌ Dynamics don't differ<br/>→ rethink approach"]

    style WIN fill:#27ae60,color:#fff
    style FAIL2 fill:#c0392b,color:#fff
```

### Metrics to compute per trace

| Metric | What it captures |
|--------|-----------------|
| **Number of hypothesis transitions** (similarity drops below threshold) | How much the model explores |
| **Mean consecutive similarity** | Overall convergence tendency |
| **Similarity at midpoint** | How committed at halfway |
| **Similarity slope** (linear fit) | Rate of convergence |
| **Max similarity drop magnitude** | Sharpness of hypothesis shifts |
| **Final convergence** (last 3 sentences similarity) | How decisively it commits |

### Per-model × per-embedding-model analysis

Report a **4×2 table** (4 reasoning models × 2 embedding models) with p-values and effect sizes. This immediately tells us:
- Which reasoning model produces most differentiable traces
- Which embedding model captures the signal best

---

## Pilot 3: Uncertainty Marker Calibration

**Core question**: Do the model's own hedging and revision signals correlate with actual errors?

> [!NOTE]
> This is **completely independent** from Pilots 1-2. Pilots 1-2 test embedding-based hypothesis detection. Pilot 3 tests linguistic uncertainty signals. They measure different things.

```mermaid
flowchart TD
    TRACES["Same traces from generation"] --> EXTRACT["Extract uncertainty markers<br/>via regex matching"]

    EXTRACT --> CATS["Categories:"]
    CATS --> HEDGE["HEDGE: 'I think', 'probably',<br/>'might be', 'perhaps', 'likely',<br/>'I'm not sure', 'it seems'"]
    CATS --> REVISE["REVISE: 'Wait', 'Actually',<br/>'No,', 'Let me redo',<br/>'that's wrong', 'correction'"]
    CATS --> VERIFY["VERIFY: 'Let me verify',<br/>'double-check', 'checking',<br/>'to confirm'"]
    CATS --> EXPLICIT["EXPLICIT: 'I'm uncertain',<br/>'not confident', 'unsure',<br/>'hard to determine'"]

    HEDGE & REVISE & VERIFY & EXPLICIT --> UMD["Compute per trace:<br/>UMD_total = markers / tokens<br/>UMD_hedge, UMD_revise, etc."]

    UMD --> CORR["Spearman(UMD, is_incorrect)<br/>per model, per category"]

    CORR --> CHECK{"Spearman > 0.15<br/>for ≥ 2 of 4 models?"}
    CHECK -->|"Yes"| PASS["✅ Markers are informative<br/>→ Use in main Exp 3"]
    CHECK -->|"No"| PARTIAL["⚠️ Drop markers<br/>→ Diagnose from embedding<br/>dynamics alone"]

    style PASS fill:#27ae60,color:#fff
    style PARTIAL fill:#f39c12,color:#fff
```

### Why per-category matters
- REVISE markers ("Wait", "Actually") may be **structural** in some models (formulaic) → low correlation
- HEDGE markers ("probably", "might") may be **genuinely calibrated** → higher correlation
- Report each category separately. If only HEDGE correlates, use only HEDGE.

---

## Execution Order

```mermaid
flowchart TD
    S1["1. Download MedQA dataset<br/>+ install deps"] --> S2["2. Sample 20 questions<br/>(stratified by difficulty)"]

    S2 --> S3["3. Generate traces<br/>(4 models × 20 Qs = 80 traces)<br/>Sequential: load model → generate all 20 → unload"]

    S3 --> S4["4. Sentence-level segmentation<br/>of all 80 traces"]

    S4 --> S5["5. PILOT 1: Embed sentences<br/>with both embedding models<br/>→ compute similarity curves<br/>→ detect hypothesis transitions<br/>→ manual validation on 10 traces"]

    S5 --> P1{"Pilot 1 Pass?"}
    P1 -->|"No"| KV["KV-cache hidden state<br/>extraction + attention entropy"]
    P1 -->|"Yes"| S6["6. PILOT 2: Compare dynamics<br/>correct vs incorrect traces<br/>→ statistical tests"]

    KV --> S6

    S6 --> P2{"Pilot 2 Pass?"}
    P2 -->|"No"| STOP["❌ Rethink"]
    P2 -->|"Yes"| S7["7. PILOT 3: Extract markers<br/>→ Spearman correlation"]

    S7 --> RESULTS["8. Compile pilot report:<br/>• Best reasoning model<br/>• Best embedding model<br/>• Marker informativeness<br/>→ Design main experiments"]

    style RESULTS fill:#27ae60,color:#fff
    style STOP fill:#c0392b,color:#fff
```

---

## A* Paper Considerations Baked Into This Design

| Principle | How We Apply It |
|-----------|----------------|
| **Clean contribution boundary** | Method = intra-trace dynamics. Not prompt engineering, not model selection. |
| **Compute advantage narrative** | 1 trace vs k=5 for SE. We measure this in wall-clock during pilots. |
| **Ablation-ready** | 4 models × 2 embeddings is already an ablation table. Shows generalizability. |
| **Cross-domain** | Same method on medical (MedQA) + math (MATH L3-5). Pilots are medical; main exps add math. |
| **Baselines set early** | Pilot results directly shape which baselines to compare against. |
| **Negative results are fine** | If smaller models (4B) don't show signal but larger ones (32B) do, that's an interesting capacity-dependent finding. |
| **Reproducibility** | Open-weight models, minimal prompts, standard datasets, all code public. |

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/config.py` | Model IDs, GPU mappings, paths, thresholds |
| `src/datasets.py` | MedQA loader + pilot sample selection |
| `src/generate_traces.py` | vLLM trace generation with `<think>` extraction |
| `src/sentence_segmenter.py` | Sentence-level trace segmentation |
| `src/embedding_pipeline.py` | BGE-M3 + Qwen3-Embedding-8B, cosine similarity |
| `src/pilot_hypothesis_detection.py` | Pilot 1: transition detection + manual validation |
| `src/pilot_dynamics.py` | Pilot 2: correct vs incorrect comparison |
| `src/pilot_markers.py` | Pilot 3: uncertainty marker extraction + Spearman |
| `src/run_pilots.py` | CLI entrypoint for all 3 pilots |
