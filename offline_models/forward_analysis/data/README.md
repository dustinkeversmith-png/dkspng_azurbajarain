# Synthetic neural telemetry data

10,000 distinct records split into 9,000 training and 1,000 evaluation records. Each JSON object uses the uploaded schema.

The activations are constructive, simulated ReLU-like telemetry and the targets and predictions are synthetic. **They are not measured executions of trained networks.** The qualitative descriptions are grounded in observed sparsity, singular-value geometry, controlled input dependence, and output error. Each sample has four text diagnoses. `eigen_signs` is provided for legacy compatibility, not as computed eigenvalue signs.

Dimensions: input 6–32, hidden width 8–48, output 1–8, batch 4–16; one or two hidden layers. Some patterns impose deliberate bottlenecks.

Files: `network_train_9000.jsonl`, `network_eval_1000.jsonl`, `manifest.json`.
