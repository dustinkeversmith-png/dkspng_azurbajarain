# Corrected synthetic telemetry (10,000 records)

- `llm_train.jsonl`: 8,000 training records spanning nine pathologies.
- `llm_validation.jsonl`: 1,000 independent in-distribution validation records spanning all nine training pathologies. Use for early stopping.
- `llm_eval_ood.jsonl`: 1,000 sign-split-cluster OOD records. Do not use for model selection.
- `manifest.json`: exact ranges, split counts, label derivation, and limitations.
- `verify.py`: independent remeasurement of eigen signs and telemetry dimensions.

The original JSONL record schema is preserved exactly. Pathology assignment is documented in manifest, not added to record schema. Inputs and activations are 4-decimal numeric values. These are **algorithmically generated** examples, not manually authored LLM examples or scientifically validated forward passes.

IMPORTANT: eigen_signs have a precise definition in manifest. Your C++ evaluator must implement the same operation or relabel these records using its OWN eigen-sign definition before training. The previous mismatch could otherwise persist.

Text deliberately avoids unsupported claims of causal attribution; it reports sample correlation. Treat rank and range descriptions as auditable observational statements. Ground truth textual consistency with a domain-specific checker is not guaranteed.
