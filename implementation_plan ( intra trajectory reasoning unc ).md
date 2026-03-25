# Intra-Trace Reasoning Uncertainty: Detecting and Repairing Premature Commitment in LLM Thinking

## Central Claim

A single reasoning trace from a thinking model (QwQ-32B, DeepSeek-R1) encodes structured epistemic signals — convergence dynamics, self-uncertainty markers, and commitment signatures — that predict answer correctness better than answer-level semantic entropy, at **1/5th the compute**. Detected failure modes enable targeted, training-free repair.

## The Paradigm Shift

```mermaid
graph LR
    subgraph "❌ Old: Multi-Trajectory Sampling"
        A1["Generate B=8 CoTs"] --> A2["Cluster across trajectories"]
        A2 --> A3["Inter-trajectory entropy"]
        A3 --> A4["Problem: B=3 is degenerate,<br/>B=8 is more expensive than SE"]
    end

    subgraph "✅ New: Single-Trace Analysis"
        B1["Generate 1 thinking trace"] --> B2["Segment at natural boundaries"]
        B2 --> B3["Embed each step"]
        B3 --> B4["Intra-trace dynamics:<br/>convergence, divergence, oscillation"]
        B4 --> B5["1 generation vs 5 for SE<br/>= genuine compute savings"]
    end

    style A4 fill:#c0392b,color:#fff
    style B5 fill:#27ae60,color:#fff
```

> [!IMPORTANT]
> **Why this works**: QwQ-32B and DeepSeek-R1 emit `<think>...</think>` blocks containing natural reasoning pivots — "Wait," "Let me reconsider," "Actually," double newlines. These are **model-defined** step boundaries, not arbitrary 128-token chunks. The model's own trace is already internally diverse: it debates, branches, backtracks, and revises within a single generation.

---

## Narrative Arc: Detect → Diagnose → Repair

```mermaid
graph TD
    subgraph "🔍 DETECT — Exp 1 + 2"
        D1["Exp 1: Answer-level SE<br/>(k=5 samples, floor)"]
        D2["Exp 2: Intra-trace dynamics<br/>(1 sample, step-level)"]
    end

    subgraph "🩺 DIAGNOSE — Exp 3"
        D3["Classify failure mode:<br/>Silent commit? Noisy struggle?<br/>Oscillating confusion?"]
    end

    subgraph "🔧 REPAIR — Exp 4"
        D4["Targeted re-prompting<br/>based on failure type"]
    end

    D1 & D2 --> D3 --> D4

    D1 -.- N1["When: how uncertain<br/>is the model?"]
    D2 -.- N2["Where: at which step<br/>does it commit?"]
    D3 -.- N3["Why: knowledge gap<br/>or knowledge suppression?"]
    D4 -.- N4["Fix: inject suppressed<br/>knowledge or decompose"]
```

---

## Design Decisions

| Decision                                                    | Rationale                                                                                                            |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Single trace, not multi-trajectory**                      | B=3 gives degenerate entropy; B=8 costs more than SE. Single trace with natural steps is cheaper AND more principled |
| **Natural boundaries, not token chunks**                    | QwQ/R1 emit reasoning pivots ("Wait", "Actually"). Model-defined, not arbitrary                                      |
| **Intra-trace similarity, not inter-trajectory clustering** | No k-means on 3 points. Cosine similarity sequence is clean and well-defined                                         |
| **QwQ-32B + DeepSeek-R1-7B**                                | Reasoning models with `<think>` traces. Fit user's compute. Cross-model validation                                   |
| **MATH Level 3-5, not GSM8K**                               | GSM8K >95% acc for 70B+ → ceiling effects. MATH has genuine difficulty                                               |
| **Drop PubMedQA**                                           | Known label quality issues                                                                                           |
| **Cluster rationales not MCQ letters**                      | MedQA is 4/5-way MCQ — NLI on letters A/B/C/D is meaningless                                                         |
| **Pilot tests FIRST**                                       | 2-3 days to validate core assumptions before 9-week commitment                                                       |

---

## Models & Infrastructure

| Component             | Choice                                     | Rationale                                                          |
| --------------------- | ------------------------------------------ | ------------------------------------------------------------------ |
| **Primary reasoning** | QwQ-32B-Preview                            | Open-weight reasoning model, `<think>` traces, fits user's compute |
| **Validation model**  | DeepSeek-R1-Distill-Qwen-7B                | Smaller reasoning model, cross-model validation                    |
| **Embeddings**        | BGE-M3 (dense mode)                        | Strong on short domain text                                        |
| **NLI**               | DeBERTa-v3-large + MedNLI-finetuned        | Answer equivalence                                                 |
| **Inference**         | vLLM                                       | Efficient generation                                               |
| **Datasets**          | MedQA (1273 test) + MATH Level 3-5 (~1500) | Medical + mathematical reasoning                                   |

> [!WARNING]
> **MedQA is MCQ.** For Exp 1 baseline, cluster the **rationale text** (the model's natural-language diagnosis), not the letter answer. This makes SE measure reasoning diversity, not just answer diversity.

---

## Experiment 1: Vanilla Semantic Entropy Baseline (Floor)

**Goal**: Establish answer-level uncertainty performance. Everything must beat this.

```mermaid
sequenceDiagram
    participant Q as Question
    participant LLM as QwQ-32B (short mode, T=0.8)
    participant NLI as DeBERTa-v3-large
    participant M as Metrics

    Q->>LLM: Generate k=5 short answers (no extended thinking)
    Note over LLM: Prompt: "Answer directly<br/>without extended explanation"

    loop For each of k=5 responses
        LLM-->>NLI: Extract rationale + answer
    end

    NLI->>NLI: Pairwise bidirectional entailment
    NLI->>NLI: Cluster equivalent rationales
    NLI-->>M: H = −Σ p(cluster) · log p(cluster)
    M->>M: AUROC (H → error), ECE, Brier
```

| Detail            | Value                                                   |
| ----------------- | ------------------------------------------------------- |
| Samples           | k=5                                                     |
| Temperature       | 0.8                                                     |
| Clustering object | Rationale text, not MCQ letter                          |
| NLI               | DeBERTa-v3-large (general) + MedNLI-finetuned (medical) |
| Datasets          | MedQA, MATH Level 3-5                                   |
| Metrics           | AUROC, ECE, Brier, wall-clock time                      |
| Expected AUROC    | ~0.65-0.72                                              |

**Compute cost**: 5 full generations per question. This is the floor.

#### [NEW] [baseline_semantic_entropy.py](file:///home/mohit_joshi/rlm%20for%20uncertanity/src/baseline_semantic_entropy.py)

---

## Experiment 2: Intra-Trace Reasoning Dynamics (Core Contribution)

**Hypothesis**: A single thinking trace contains step-level dynamics that predict correctness — incorrect reasoning exhibits **premature convergence** while correct reasoning maintains **productive divergence** before converging.

### Step 1: Natural Boundary Segmentation

```mermaid
flowchart TD
    TRACE["Raw <think> trace from QwQ-32B<br/>(typically 500-3000 tokens)"] --> PARSE["Parse natural boundaries"]

    PARSE --> SIG["Boundary signals:"]
    SIG --> S1["'Wait,' / 'Wait, '"]
    SIG --> S2["'Actually,' / 'Actually, '"]
    SIG --> S3["'Let me reconsider'"]
    SIG --> S4["'Hmm,' / 'Hmm, '"]
    SIG --> S5["'No,' / 'No, that's wrong'"]
    SIG --> S6["'So,' (preceded by reasoning)"]
    SIG --> S7["Double newlines / paragraph breaks"]

    S1 & S2 & S3 & S4 & S5 & S6 & S7 --> STEPS["Segmented Steps: s₁, s₂, ..., s_N"]
    STEPS --> FILTER["Filter: drop steps < 10 tokens"]
    FILTER --> FINAL["Final: S = [s₁, s₂, ..., s_M]"]

    style FINAL fill:#2c3e50,color:#fff
```

### Step 2: Step Embedding & Dynamics

```mermaid
flowchart TD
    STEPS["Steps s₁, s₂, ..., s_M"] --> EMB["BGE-M3 dense embedding<br/>→ e₁, e₂, ..., e_M"]

    EMB --> THREE["Three Intra-Trace Metrics"]

    THREE --> SIM["① Consecutive Similarity<br/>sim_d = cos(e_d, e_{d+1})<br/>→ similarity curve over depth"]
    THREE --> VAR["② Step Variance<br/>σ²_d = Var(e₁...e_d)<br/>→ accumulated diversity"]
    THREE --> DRIFT["③ Commitment Drift<br/>drift_d = cos(e_d, e_M)<br/>→ how similar to final answer?"]

    SIM --> CURVE1["Convergence curve"]
    VAR --> CURVE2["Diversity curve"]
    DRIFT --> CURVE3["Drift-to-conclusion curve"]

    CURVE1 & CURVE2 & CURVE3 --> SIG2["Commitment Signature C =<br/>first d where sim_d > τ<br/>AND stays above τ"]

    SIG2 --> PRED["Does C predict correctness?<br/>Early C = premature commit<br/>= more likely wrong"]

    style SIG2 fill:#e74c3c,color:#fff
    style PRED fill:#27ae60,color:#fff
```

### What Gets Measured

| Question                                                        | How                                     |
| --------------------------------------------------------------- | --------------------------------------- |
| Does C correlate with correctness?                              | AUROC of C → error                      |
| Do incorrect traces converge faster?                            | Mann-Whitney on mean sim_d              |
| Does similarity curve shape predict better than answer-level H? | Compare AUROC: curve features vs. Exp 1 |
| Compute advantage?                                              | 1 gen vs 5 → wall-clock ratio           |

**Hero visualization**: Similarity curves over depth, colored by correctness (correct = blue, stays low then rises; incorrect = red, rises early).

### Key Difference from Prior Work

| Aspect               | Semantic Entropy                | Ours                             |
| -------------------- | ------------------------------- | -------------------------------- |
| Input                | k=5 separate generations        | **1 thinking trace**             |
| Measures             | Answer diversity across samples | Step dynamics within one trace   |
| Cost                 | 5× generation                   | **1× generation**                |
| Commitment detection | Not possible                    | **Yes — commitment signature C** |

#### [NEW] [trace_segmenter.py](file:///home/mohit_joshi/rlm%20for%20uncertanity/src/trace_segmenter.py)

#### [NEW] [trace_dynamics.py](file:///home/mohit_joshi/rlm%20for%20uncertanity/src/trace_dynamics.py)

#### [NEW] [commitment_signature.py](file:///home/mohit_joshi/rlm%20for%20uncertanity/src/commitment_signature.py)

---

## Experiment 3: Failure Mode Diagnosis

**Goal**: Not just _that_ the model is wrong, but _why_ — which determines what repair works.

### Self-Uncertainty Markers (Free Signal)

```mermaid
flowchart TD
    TRACE["Parsed thinking trace"] --> EXTRACT["Extract self-uncertainty markers"]

    EXTRACT --> M1["Hedging: 'I think', 'probably',<br/>'might be', 'I'm not sure'"]
    EXTRACT --> M2["Revision: 'Wait', 'Actually',<br/>'No, that's wrong'"]
    EXTRACT --> M3["Explicit: 'I'm uncertain',<br/>'not confident'"]
    EXTRACT --> M4["Verification: 'Let me verify',<br/>'double-check'"]

    M1 & M2 & M3 & M4 --> UMD["UMD_d = markers / tokens<br/>at step d"]
```

### Three Failure Modes

```mermaid
flowchart TD
    CS["Commitment Signature C"] --> CLASS{"Classify"}
    UMD["Uncertainty Marker Density"] --> CLASS

    CLASS --> FM1["🔴 EARLY SILENT COMMIT<br/>C early + UMD low<br/><br/>Commits fast, no hedging.<br/>→ Knowledge suppression"]

    CLASS --> FM2["🟡 LATE NOISY COMMIT<br/>C late + UMD high<br/><br/>Struggles, eventually settles.<br/>→ Genuine difficulty"]

    CLASS --> FM3["🟠 OSCILLATING<br/>C undefined + UMD high<br/><br/>Never settles.<br/>→ Outside knowledge boundary"]

    FM1 --> R1["→ Repair A: Elicit<br/>suppressed knowledge"]
    FM2 --> R2["→ Repair B: Contrastive<br/>lesson from trace"]
    FM3 --> R3["→ Repair C: Decompose"]

    style FM1 fill:#c0392b,color:#fff
    style FM2 fill:#f39c12,color:#fff
    style FM3 fill:#e67e22,color:#fff
```

| Metric                                               | Method                                            |
| ---------------------------------------------------- | ------------------------------------------------- |
| Do failure modes map to different accuracy profiles? | Accuracy within each mode                         |
| Is UMD at commitment point predictive?               | Spearman(UMD_C, correctness)                      |
| Domain differences?                                  | MedQA (expect more N) vs MATH (expect more LNC) |

#### [NEW] [uncertainty_markers.py](file:///home/mohit_joshi/rlm%20for%20uncertanity/src/uncertainty_markers.py)

#### [NEW] [failure_modes.py](file:///home/mohit_joshi/rlm%20for%20uncertanity/src/failure_modes.py)

---

## Experiment 4: Targeted Step-Level Repair (The Payoff)

**Repositioned claim**: "Intra-trace uncertainty signals are better selectors and teachers than answer-level signals for inference-time reasoning improvement."

```mermaid
flowchart TD
    FM["Diagnosed Failure Mode"] --> BRANCH{"Which type?"}

    BRANCH -->|"🔴 Early Silent"| ESC["REPAIR A: Latent Knowledge"]
    BRANCH -->|"🟡 Late Noisy"| LNC["REPAIR B: Contrastive Lesson"]
    BRANCH -->|"🟠 Oscillating"| ONC["REPAIR C: Decompose"]

    subgraph "Repair A: Knowledge Injection"
        ESC --> KE["Fresh context query:<br/>'Key facts about [topic]?'"]
        KE --> FLAT["F_latent = elicited facts"]
        FLAT --> NLI_CHECK["NLI: F_latent contradicts<br/>committed claim?"]
        NLI_CHECK -->|"Yes"| INJECT["Re-run with:<br/>'Note these facts: [F_latent]'"]
        NLI_CHECK -->|"No"| FALL["→ Fall to Repair B"]
    end

    subgraph "Repair B: Trace Self-Contrast"
        LNC --> COMPARE["Extract model's OWN<br/>abandoned branches from trace:<br/>'I initially thought X but...'"]
        COMPARE --> LESSONS["Lesson: 'You considered X<br/>but abandoned it. Evidence<br/>for X was: [...]'"]
        LESSONS --> RERUN["Re-run with lesson prefix"]
    end

    subgraph "Repair C: Decompose"
        ONC --> DECOMP["Break into 2-3<br/>sub-questions"]
        DECOMP --> SUB["Solve independently"]
        SUB --> SYNTH["Synthesize"]
    end

    INJECT --> EVAL["Accuracy improvement"]
    FALL --> COMPARE
    RERUN --> EVAL
    SYNTH --> EVAL

    style INJECT fill:#2980b9,color:#fff
    style RERUN fill:#9b59b6,color:#fff
    style SYNTH fill:#e67e22,color:#fff
```

### Why This Is NOT Reflexion

|                              | Reflexion                | Ours                                     |
| ---------------------------- | ------------------------ | ---------------------------------------- |
| Feedback                     | Oracle (ground truth)    | Intra-trace signals (no labels)          |
| Compared                     | Output vs correct answer | Model's own abandoned branches vs commit |
| Lesson type                  | "I got X wrong"          | "You considered Y but abandoned it"      |
| Works without correct trace? | No                       | **Yes**                                  |

### Baselines

| Baseline                                   | Controls for                  |
| ------------------------------------------ | ----------------------------- |
| Vanilla re-run (different seed)            | Stochasticity                 |
| Best-of-5 self-consistency                 | Multi-sample selection        |
| Reflexion (oracle feedback)                | Oracle-label intervention     |
| Undifferentiated repair (Repair A for all) | Value of failure-mode routing |

> [!IMPORTANT]
> **Table 3 design**: Show repair improves accuracy most on ESC subset, where suppressed knowledge injection targets the exact failure mechanism. Reflexion can't match this because it doesn't know _why_ the model failed.

#### [NEW] [repair_pipeline.py](file:///home/mohit_joshi/rlm%20for%20uncertanity/src/repair_pipeline.py)

#### [NEW] [latent_knowledge.py](file:///home/mohit_joshi/rlm%20for%20uncertanity/src/latent_knowledge.py)

---

## Known Drawbacks & Mitigations

| #   | Drawback                                            | Severity | Mitigation                                                                |
| --- | --------------------------------------------------- | -------- | ------------------------------------------------------------------------- |
| 1   | Natural boundaries may be rhetorical, not epistemic | High     | Pilot Test 1 validates; fallback: hidden-state cosine drops               |
| 2   | Self-uncertainty markers uncalibrated               | High     | Pilot Test 3; if Spearman < 0.15, drop markers, diagnose from C alone     |
| 3   | Single trace = no inter-sample diversity            | Medium   | Trace contains abandoned branches; Exp 1 provides inter-sample comparison |
| 4   | BGE-M3 may not separate short steps                 | High     | Pilot Test 2; fallback: model's own hidden states                         |
| 5   | Commitment threshold τ is arbitrary                 | Medium   | Calibrate on 10% held-out; sensitivity over τ ∈ {0.7-0.9}                 |
| 6   | Results may not generalize beyond QwQ/R1            | Medium   | Validate on both models; explicit limitation                              |
| 7   | F_latent in Repair A may hallucinate                | Medium   | T=0.0 greedy; NLI contradiction acts as filter                            |
| 8   | Not all traces contain abandoned branches           | Medium   | Measure fraction; no-branch traces fall to Repair C                       |

---

## Pilot Validation Tests (Week 1 — Run FIRST)

> [!CAUTION]
> 2-3 days total. If any fails, redesign before investing 9 weeks.

### Pilot 1: Trace Structure

```mermaid
flowchart LR
    A["20 MedQA + 20 MATH<br/>traces from QwQ-32B"] --> B["Parse boundaries"]
    B --> C{"Mean steps ≥ 5?"}
    C -->|Yes| D["✅ Proceed"]
    C -->|No| E["❌ Use hidden-state drops"]

    style D fill:#27ae60,color:#fff
    style E fill:#c0392b,color:#fff
```

### Pilot 2: Embedding Signal

```mermaid
flowchart LR
    A["40 traces (20 correct,<br/>20 incorrect)"] --> B["Embed steps, compute<br/>similarity curves"]
    B --> C{"Mann-Whitney p < 0.05<br/>correct vs incorrect?"}
    C -->|Yes| D["✅ Signal exists"]
    C -->|No| E["❌ Try model hidden states"]

    style D fill:#27ae60,color:#fff
    style E fill:#c0392b,color:#fff
```

### Pilot 3: Marker Calibration

```mermaid
flowchart LR
    A["Same 40 traces"] --> B["Extract markers,<br/>compute UMD"]
    B --> C{"Spearman(UMD, error) > 0.15?"}
    C -->|Yes| D["✅ Use in Exp 3"]
    C -->|No| E["⚠️ Drop markers,<br/>use C alone"]

    style D fill:#27ae60,color:#fff
    style E fill:#e67e22,color:#fff
```

---

## Timeline

```mermaid
gantt
    title 9-Week Timeline
    dateFormat YYYY-MM-DD
    axisFormat %b %d

    section Week 1: Pilots
    Setup vLLM + QwQ-32B          :a1, 2026-02-23, 3d
    Pilot Tests 1-3               :crit, a2, after a1, 4d

    section Week 2-3: Core
    Exp 1 Baseline SE             :a3, after a2, 5d
    Exp 2 Trace dynamics + C      :a4, after a3, 9d

    section Week 4-5: Diagnosis
    Exp 3 Markers + failure modes :a5, after a4, 7d
    Cross-model validation        :a6, after a5, 7d

    section Week 6-7: Repair
    Exp 4 All repairs + baselines :a7, after a6, 14d

    section Week 8-9: Paper
    Results + figures + writing   :a8, after a7, 14d
```

---

## Positioning Against Prior Work

| Prior Work                            | How we differ                                              |
| ------------------------------------- | ---------------------------------------------------------- |
| Semantic Entropy (Kuhn et al.)        | 1 trace vs k=5 samples; step dynamics not answer diversity |
| Reflexion (Shinn et al.)              | No oracle; repair from model's own trace structure         |
| Self-Consistency (Wang et al.)        | Analyze process, not just answers                          |
| Self-Refine (Madaan et al.)           | Diagnose _why_ then match repair type                      |
| Verbalized uncertainty (Xiong et al.) | Implicit signals (convergence), not explicit self-reports  |

---

## Files to Implement

| File                           | Experiment                       |
| ------------------------------ | -------------------------------- |
| `config.py`                    | All — hyperparameters, paths     |
| `model_manager.py`             | All — vLLM, `<think>` extraction |
| `datasets.py`                  | All — MedQA + MATH loaders       |
| `baseline_semantic_entropy.py` | Exp 1                            |
| `trace_segmenter.py`           | Exp 2 — boundary parsing         |
| `trace_dynamics.py`            | Exp 2 — sim, variance, drift     |
| `commitment_signature.py`      | Exp 2 — C detection              |
| `uncertainty_markers.py`       | Exp 3 — marker extraction        |
| `failure_modes.py`             | Exp 3 — ESC/LNC/ONC              |
| `repair_pipeline.py`           | Exp 4 — routed repair            |
| `latent_knowledge.py`          | Exp 4 — Repair A                 |
| `metrics.py`                   | All — AUROC, ECE, Spearman       |
| `run_pilots.py`                | Pilots 1-3                       |
| `run_experiment.py`            | Main CLI                         |
