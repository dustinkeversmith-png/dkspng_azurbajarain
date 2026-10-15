# Corrected Synthetic Telemetry Dataset (10,000 Records)

- `llm_train.jsonl`: 8,000 training records spanning nine mathematical pathologies (`rank_one`, `dead_relu`, `half_space_clipping`, `range_attenuation`, `feature_axis_dominance`, `two_cluster`, `bottleneck`, `balanced_projection`, `output_bias`).
- `llm_validation.jsonl`: 1,000 independent in-distribution validation records spanning all nine training pathologies. Used for early stopping and model selection.
- `llm_eval_ood.jsonl`: 1,000 out-of-distribution (`sign_split_clusters`) evaluation records. Held out for final generalization evaluation.
- `network_eval_1000.jsonl`: 1,000 supplementary evaluation records using qualitative singular-energy and mean-residual prose templates.
- `manifest.json`: Exact parameter ranges, split counts, pathology coverage, `eigen_sign_definition`, and dataset caveats.
- `verify.py`: Independent verification script for `eigen_signs` and telemetry tensor dimensions.

## Alignment with the Refactored C++ Architecture

1. **`eigen_signs` Definition**:
   `ObservationExtractor::extract_input_eigen_signs`, `SpectralFeatureExtractor`, and `MultiFamilyTokenExtractor` implement the exact `manifest.json` specification: the signs ($\pm 1$, with $0 \mapsto +1$) of the 4 descending eigenvalues of the $4 \times 4$ symmetric zero-diagonal sample covariance matrix of the first 4 centered input dimensions (divisor $\max(1, N-1)$).
2. **`target_text` Consistency**:
   All four fields (`pathway_activity`, `feature_sensitivity`, `representational_structure`, `functional_mapping`) in `llm_train.jsonl`, `llm_validation.jsonl`, and `llm_eval_ood.jsonl` match `BatchStatementChecker` (`NarrationStyle::kBatchStatistics`, `SlotBinding::kBatchStatistics`) with 100.00% ground-truth consistency.
