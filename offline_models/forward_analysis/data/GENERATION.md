# Telemetry & Ground-Truth Target Generation Reference

## 1. Overview of Telemetry Synthesis

The `offline_models/forward_analysis/data/` splits (`llm_train.jsonl`, `llm_validation.jsonl`, `llm_eval_ood.jsonl`) contain self-consistent multi-dimensional forward-pass telemetry paired with verifiable mathematical diagnostics across four observation fields (`pathway_activity`, `feature_sensitivity`, `representational_structure`, `functional_mapping`) and 4-dimensional spectral `eigen_signs`.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   TELEMETRY & DIAGNOSTIC SYNTHESIZER                   │
│                                                                        │
│  1. Samples network topology: D_in in [6, 32], L in {1, 2},            │
│     W_l in [8, 48], D_out in [1, 8], batch rows N in [4, 16]           │
│  2. Constructs inputs, layer_activations, predictions, and targets     │
│     exhibiting a specified mathematical pathology                      │
│  3. Computes exact batch invariants (dead units, nonpositive rows,     │
│     centered numerical rank > 1e-3, input-hidden correlations,         │
│     output range ratios in [0.8, 1.2], sign agreement, eigen_signs)    │
│  4. Emits the JSONL record for DualObjectiveDistillationBridge         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               DOMAIN-AWARE DISTILLER TRAINING & EVALUATION             │
│                                                                        │
│  • Extracts 5 token families (Channel, Sample, Relationship, Layer,    │
│    Global) across 6 mathematical domains                               │
│  • Trains 8 domain prediction heads + PerceiverIOEncoder +             │
│    AutoregressiveCausalDecoder with delexicalized slot tokens          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. JSONL Record Schema

Every line in `llm_train.jsonl`, `llm_validation.jsonl`, and `llm_eval_ood.jsonl` is a single JSON object with the following structure:

```json
{
  "sample_id": "syn_math_train_00001",
  "telemetry": {
    "inputs": [
      [1.861, -1.4616, -0.2565, "... (D_in floats per row, N rows)"]
    ],
    "targets": [
      [3.07],
      [1.5589]
    ],
    "predictions": [
      [3.0152],
      [2.1612]
    ],
    "layer_activations": [
      [
        [3.9377, 3.0287, "... (W_1 floats per row, N rows)"]
      ],
      [
        [2.3402, 1.6306, "... (W_2 floats per row, N rows, if L = 2)"]
      ]
    ]
  },
  "ground_truth_targets": {
    "eigen_signs": [1, 1, -1, -1],
    "target_text": {
      "pathway_activity": "Across 14 observations, the hidden widths are 27 and 14. Layer 1 has 0 coordinates that remain zero throughout the batch and 0 observations with no positive coordinates. Layer 2 has 0 coordinates that remain zero throughout the batch and 0 observations with no positive coordinates.",
      "feature_sensitivity": "Observed association only: input coordinates 0 and 1 have the strongest absolute sample correlations with at least one final hidden coordinate; coordinates 19 and 23 have the weakest. These associations do not establish causal feature influence.",
      "representational_structure": "The centered activation matrices have observed numerical ranks 1, 1 across successive hidden layers. This is a narrow observed representation relative to its hidden widths.",
      "functional_mapping": "Comparing predictions against targets over 1 output coordinates, 0 have a smaller prediction range, 0 have a larger range, and 1 have broadly comparable ranges using a range-ratio window from 0.8 to 1.2. Prediction and target signs agree in 14 of 14 entries. These statements describe only the observed batch."
    }
  }
}
```

---

## 3. Exact Mathematical Definitions of Target Quantities

1. **`eigen_signs` (`[4 ints in {+1, -1}]`)**:
   * Select the first $K = \min(4, D_{\text{in}})$ input columns $X_{:, 1:K} \in \mathbb{R}^{N \times K}$.
   * Center each column across the $N$ batch observations: $\tilde{X} = X_{:, 1:K} - \mathbf{1}\bar{x}^T$.
   * Form the sample covariance matrix $C = \frac{1}{\max(1, N-1)} \tilde{X}^T \tilde{X}$ and set $\operatorname{diag}(C) = 0$.
   * Compute the $K$ real eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge \lambda_4$ in descending order and emit $+1$ if $\lambda_k \ge 0$ and $-1$ if $\lambda_k < 0$.
2. **`pathway_activity`**:
   * Reports batch observation count $N$, hidden layer widths $W_1$ (and $W_2$ when $L=2$), number of coordinates per layer whose absolute activation never exceeds `dead_activation_threshold` ($10^{-6}$) across the batch, and number of observations per layer with no positive coordinates ($\max_u a_{n,u} \le 0$).
3. **`feature_sensitivity`**:
   * For each input coordinate $c \in \{0, \dots, D_{\text{in}}-1\}$, computes $s_c = \max_{u} |\operatorname{corr}(X_{:, c}, H_{L, :, u})|$ against the final hidden layer $H_L$.
   * If $\max_c s_c \le 10^{-6}$, emits that all final hidden coordinates are constant; otherwise reports the two input coordinates with the largest $s_c$ (tie-breaking by smaller coordinate index) and the two with the smallest $s_c$.
4. **`representational_structure`**:
   * For each hidden layer $\ell \in \{1, \dots, L\}$, centers $H_\ell$ across the $N$ rows and counts singular values strictly greater than `numerical_rank_tolerance` ($10^{-3}$).
   * Classifies the representation as narrow (*"This is a narrow observed representation relative to its hidden widths."*) if every layer has numerical rank $\le 2$ (`narrow_rank_limit`), and multidirectional (*"The observed activation variation occupies multiple directions in at least one layer."*) otherwise.
5. **`functional_mapping`**:
   * For each output coordinate $k \in \{0, \dots, D_{\text{out}}-1\}$, computes the range ratio $r_k = \frac{\max_n \hat{Y}_{n,k} - \min_n \hat{Y}_{n,k}}{\max_n Y_{n,k} - \min_n Y_{n,k}}$ and counts outputs with smaller ($r_k < 0.8$), larger ($r_k > 1.2$), and comparable ($0.8 \le r_k \le 1.2$) ranges, along with total entry-wise sign agreements across all $N \times D_{\text{out}}$ entries.
