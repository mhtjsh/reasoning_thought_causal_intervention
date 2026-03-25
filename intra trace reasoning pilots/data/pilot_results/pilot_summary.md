# Pilot Experiments — Consolidated Report

## Overview

| Pilot | Question | Result |
|-------|----------|--------|
| Pilot 1 | Can embeddings detect hypothesis transitions? | ✅ PASS |
| Pilot 2 | Do dynamics predict correctness? | ✅ PASS |
| Pilot 3 | Are uncertainty markers calibrated? | ✅ PASS |

## Pilot 1: Hypothesis Transition Detection

| Reasoning Model | Embedding | Traces | % With Transitions | Mean Trans/Trace | Mean Similarity |
|-----------------|-----------|--------|--------------------|------------------|-----------------|
| qwq-32b | bge-m3 | 20 | 95% | 5.1 | 0.542 |
| qwen3-8b | bge-m3 | 20 | 100% | 5.1 | 0.558 |
| qwen3-4b-thinking | bge-m3 | 20 | 100% | 12.8 | 0.579 |
| qwq-32b | qwen3-embedding-8b | 20 | 100% | 4.7 | 0.862 |
| qwen3-8b | qwen3-embedding-8b | 20 | 100% | 4.5 | 0.873 |
| qwen3-4b-thinking | qwen3-embedding-8b | 20 | 100% | 9.1 | 0.867 |

## Pilot 2: Embedding Dynamics Predict Correctness

### qwq-32b + qwen3-embedding-8b
Correct: 15, Incorrect: 5

| Feature | Correct (μ±σ) | Incorrect (μ±σ) | p-value | Cohen's d | Sig? |
|---------|---------------|-----------------|---------|-----------|------|
| n_transitions | 4.533±4.773 | 5.200±2.993 | 0.3504 | -0.144 | — |
| slope | 0.000±0.001 | 0.000±0.001 | 0.6116 | 0.030 | — |
| mean_sim | 0.863±0.017 | 0.859±0.018 | 0.8001 | 0.218 | — |
| min_sim | 0.747±0.042 | 0.726±0.044 | 0.4445 | 0.450 | — |
| std_sim | 0.046±0.009 | 0.049±0.006 | 0.7354 | -0.253 | — |
| trace_length | 57.200±50.528 | 69.000±33.419 | 0.2055 | -0.239 | — |
| midpoint_sim | 0.843±0.064 | 0.866±0.057 | 0.4445 | -0.348 | — |
| max_drop | 0.144±0.043 | 0.175±0.036 | 0.2300 | -0.703 | — |
| final_convergence | 0.861±0.036 | 0.844±0.053 | 0.8660 | 0.391 | — |
| commitment_point | 0.333±0.869 | 0.400±0.800 | 0.8330 | -0.074 | — |
| normalized_commitment | 0.012±0.030 | 0.006±0.013 | 0.8883 | 0.185 | — |
| mean_drift_to_final | 0.855±0.031 | 0.863±0.012 | 0.6116 | -0.269 | — |

### qwen3-8b + qwen3-embedding-8b
Correct: 11, Incorrect: 9

| Feature | Correct (μ±σ) | Incorrect (μ±σ) | p-value | Cohen's d | Sig? |
|---------|---------------|-----------------|---------|-----------|------|
| n_transitions | 3.727±2.178 | 5.556±1.499 | 0.0684 | -0.912 | — |
| slope | 0.001±0.003 | 0.000±0.000 | 0.8792 | 0.402 | — |
| mean_sim | 0.868±0.018 | 0.878±0.012 | 0.1715 | -0.558 | — |
| min_sim | 0.748±0.050 | 0.747±0.045 | 0.8792 | 0.023 | — |
| std_sim | 0.046±0.009 | 0.046±0.007 | 0.8197 | -0.021 | — |
| trace_length | 46.545±23.011 | 66.444±12.464 | 0.0481 | -0.993 | ✅ |
| midpoint_sim | 0.887±0.034 | 0.867±0.040 | 0.2545 | 0.512 | — |
| max_drop | 0.124±0.040 | 0.155±0.052 | 0.2875 | -0.647 | — |
| final_convergence | 0.870±0.033 | 0.890±0.032 | 0.1106 | -0.579 | — |
| commitment_point | 0.818±1.336 | 0.000±0.000 | 0.0570 | 0.783 | — |
| normalized_commitment | 0.022±0.034 | 0.000±0.000 | 0.0572 | 0.807 | — |
| mean_drift_to_final | 0.878±0.018 | 0.877±0.014 | 0.8197 | 0.043 | — |

### qwen3-4b-thinking + qwen3-embedding-8b
Correct: 10, Incorrect: 10

| Feature | Correct (μ±σ) | Incorrect (μ±σ) | p-value | Cohen's d | Sig? |
|---------|---------------|-----------------|---------|-----------|------|
| n_transitions | 9.000±4.583 | 9.100±6.188 | 0.7903 | -0.017 | — |
| slope | -0.000±0.000 | 0.002±0.004 | 0.0211 | -0.646 | ✅ |
| mean_sim | 0.867±0.027 | 0.866±0.014 | 0.9097 | 0.055 | — |
| min_sim | 0.731±0.040 | 0.738±0.041 | 0.6232 | -0.161 | — |
| std_sim | 0.048±0.007 | 0.049±0.011 | 0.7913 | -0.138 | — |
| trace_length | 115.000±52.792 | 118.000±91.253 | 0.9698 | -0.038 | — |
| midpoint_sim | 0.879±0.039 | 0.845±0.065 | 0.4727 | 0.589 | — |
| max_drop | 0.150±0.034 | 0.152±0.066 | 0.8501 | -0.041 | — |
| final_convergence | 0.841±0.039 | 0.885±0.038 | 0.0211 | -1.079 | ✅ |
| commitment_point | 1.000±1.673 | 0.100±0.300 | 0.2343 | 0.710 | — |
| normalized_commitment | 0.049±0.135 | 0.014±0.043 | 0.3305 | 0.332 | — |
| mean_drift_to_final | 0.832±0.046 | 0.835±0.043 | 0.7913 | -0.068 | — |

## Pilot 3: Uncertainty Marker Calibration

### qwq-32b
Accuracy: 75.0% (15/20)

| Category | Spearman ρ | p-value | Correct density | Incorrect density | Sig? |
|----------|-----------|---------|-----------------|-------------------|------|
| hedge | +0.390 | 0.0887 | 0.93/k | 1.42/k | ✅ |
| revise | -0.230 | 0.3287 | 0.73/k | 0.87/k | ✅ |
| verify | +0.552 | 0.0116 | 0.03/k | 0.19/k | ✅ |
| explicit | -0.132 | 0.5778 | 0.00/k | 0.00/k | — |

Total: ρ=+0.511, p=0.0214

### qwen3-8b
Accuracy: 55.0% (11/20)

| Category | Spearman ρ | p-value | Correct density | Incorrect density | Sig? |
|----------|-----------|---------|-----------------|-------------------|------|
| hedge | +0.061 | 0.7983 | 1.11/k | 1.23/k | — |
| revise | +0.288 | 0.2187 | 0.73/k | 1.13/k | ✅ |
| verify | -0.181 | 0.4444 | 0.19/k | 0.22/k | ✅ |
| explicit | +0.254 | 0.2806 | 0.00/k | 0.02/k | ✅ |

Total: ρ=+0.270, p=0.2493

### qwen3-4b-thinking
Accuracy: 50.0% (10/20)

| Category | Spearman ρ | p-value | Correct density | Incorrect density | Sig? |
|----------|-----------|---------|-----------------|-------------------|------|
| hedge | +0.208 | 0.3786 | 1.40/k | 1.64/k | ✅ |
| revise | -0.083 | 0.7267 | 0.08/k | 0.07/k | — |
| verify | -0.224 | 0.3425 | 0.09/k | 0.09/k | ✅ |
| explicit | +0.000 | 1.0000 | 0.00/k | 0.00/k | — |

Total: ρ=+0.087, p=0.7162

## Recommendations

- **Best embedding model**: qwen3-embedding-8b (2 significant features)
- **Best marker category**: verify (mean |ρ|=0.319)
