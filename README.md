# stevejob

Tiny neural workflows for Azure Functions. You describe a network as a C++ `NeuralNetwork`. The durable loop asks the LLM to edit one slice of that network per trial. The architecture scores the full network and writes the checkpoint only when the loss drops.

Python passes blob names and the slice text. Numeric work, HMAC signing, HTTP to Azurite, and checkpoint parsing run in the compiled extension `build/native/tensor_engine`. The one Python walk that stays is `tools/core/host.py`, which discovers tool classes for the Functions host.

## Create a model

A network is a chain of layers. Each layer has a name, an input width, an output width, and an activation. The next layer's input width must equal the previous layer's output width. The architecture that adopts the network owns the loss. A seeded custom stack is scored by the regressor.

```python
from tools import tensor_engine

model = tensor_engine.NeuralNetwork("CustomDeepClassifier")
model.add_layer("stem", 8, 16, tensor_engine.ReLUActivation()) \
     .add_layer("hidden", 16, 8, tensor_engine.TanhActivation()) \
     .add_layer("head", 8, 1, tensor_engine.LinearActivation())

model.initialize_default_weights()
receipt = model.seed_to_storage("custom_deep.json", "custom_deep_journal.json")
```

`initialize_default_weights` fills each weight with `0.02 * (row - col)` and leaves each bias at `0.1`. `seed_to_storage` scores the network, packs it as an SJC1 blob in the `models` container, and writes the first journal entry under `evals`. The receipt contains `status`, `loss`, `max_residual`, `checksum`, and `layers`. It does not contain weight matrices.

Activations are `LinearActivation`, `ReLUActivation`, `TanhActivation`, and `SigmoidActivation`. Each one exposes `apply` and `derivative`. Hidden layers use the activation you set. The output layer is linear when you pass `LinearActivation()`, which is what you want for regression. The activation travels with the checkpoint, so a Tanh hidden layer is still Tanh after the blob is loaded again.

A shape list still works when every hidden layer is ReLU and the last layer is linear:

```python
tensor_engine.seed_stack_native([[2, 4], [4, 1]], "toy.json", "toy_journal.json")
tensor_engine.seed_stack_native([[4, 8], [8, 4], [4, 1]], "deep.json", "deep_journal.json")
tensor_engine.seed_stack_native([[8, 16], [16, 1]], "wide.json", "wide_journal.json")
```

## How a trial reaches the LLM

`optimize_modular_orchestrator` is the training loop. One HTTP POST runs every trial. The orchestrator does not send the whole network.

1. `read_model_meta` loads the SJC1 blob and returns the slice names plus the current loss.
2. The loop picks the next slice in round-robin order: trial `n` uses slice `n % slice_count`.
3. `read_layer_slice` encodes that block as a TOTR string and a short bias string.
4. `activity_llm_synthesize_slice` sends one user message to Azure OpenAI and parses the JSON reply into `w` and `b`.
5. `commit_slice_native` decodes those strings, writes them into that block, and scores the full network.

TOTR is a rank-1 int8 token:

```text
r:1|g:<gamma>|A:<row ints>|B:<col ints>
```

C++ rebuilds the block with `W[row][col] = (A[row] * gamma) * (B[col] * gamma)`. Integers outside `-99` to `99` cannot be decoded. The tournament parser clamps a code onto that edge before scoring, and the prompt prints the arm's `int_bound` (default `-25` to `25`).

The commit gate:

| Result | Status | Blob |
| --- | --- | --- |
| Loss above 100, or the token cannot be decoded | `rejected_divergent` | unchanged |
| Loss greater than or equal to the stored loss | `rejected` | unchanged |
| Loss strictly lower | `committed_natively` | replaced, journal appended |

The journal keeps the last five committed scores. The target the network is fit against stays inside the C++ scorer. The prompt never includes that formula. For a 2-input network the grid matches `sin(x0) + cos(x1)` over 50 points. Wider inputs extend that same alternating sin/cos sum. Output width 1 is the intended regression head.

Start the host, then post a run. Set `PYTHONPATH` to the virtualenv site-packages so the worker can import the agent packages:

```powershell
$env:PYTHONPATH = "C:\Users\Cutie Magic 500\stevejob\stevejob\.venv\Lib\site-packages"
func start
```

```powershell
Invoke-RestMethod -Uri "http://localhost:7071/api/orchestrators/optimize_modular_orchestrator" `
  -Method Post `
  -ContentType "application/json" `
  -Body (@{
    max_trials = 30
    target_loss = 0.05
    checkpoint_blob = "custom_deep.json"
    journal_blob = "custom_deep_journal.json"
  } | ConvertTo-Json)
```

Defaults when those fields are omitted: 30 trials, target loss `0.05`, blobs `modular_model.json` and `modular_journal.json`. The loop stops early once the best committed loss is at or below the target. Poll `statusQueryGetUri` from the 202 response until `runtimeStatus` is `Completed` or `Failed`. The response body includes a management key. Do not log that query string.

The returned history is one row per trial: `trial`, `layer`, `loss`, `status`. `best_loss` updates only on `committed_natively`.

## Tournament loop

`modular_model_orchestrator` is the expanded-budget loop. C++ still owns the tensors, the score, and the write. The completion stays one stateless user message. Each trial asks for three candidates, and `commit_tournament` scores all three in C++. It writes only the lowest finite loss, and only when that loss is lower than the stored loss by more than `1e-4`.

Two concrete models ship with their own directives:

```python
tensor_engine.MlpRegressor().seed_to_storage("mlp.json", "mlp_journal.json", 10)
tensor_engine.BottleneckAutoencoder().seed_to_storage("ae.json", "ae_journal.json", 10)
```

`MlpRegressor` is 4-8-4-1, ReLU hidden layers and a linear head, scored as regression MSE. `BottleneckAutoencoder` is 8-2-8, a ReLU encoder and a linear decoder, scored as reconstruction MSE. Each model returns a task, a model type, and search rules. `read_architecture_slice` adds a finite-difference gradient hint (`A:[-1, +2] B:[0, +1]`, comma-separated `+2`, `+1`, `0`, `-1`, `-2`) and telemetry (`condition`, `sparsity`, `variance`). Those values are scalars. Weight matrices stay in the blob.

The same HTTP starter accepts the orchestrator name:

```powershell
Invoke-RestMethod -Uri "http://localhost:7071/api/orchestrators/modular_model_orchestrator" `
  -Method Post `
  -ContentType "application/json" `
  -Body (@{
    max_trials = 40
    target_loss = 0.01
    checkpoint_blob = "mlp.json"
    journal_blob = "mlp_journal.json"
    journal_capacity = 10
    rolling_config = @{ visible_history_count = 4; tracked_metrics = @("loss", "delta", "status") }
    training_params = @{ target_sparsity = 0.20; exploration_mode = "tournament_triad" }
  } | ConvertTo-Json)
```

History rows from this loop also include `delta` and `winner`.

## Current configuration

| Setting | Value |
| --- | --- |
| Single-slice orchestrator | `optimize_modular_orchestrator` |
| Tournament orchestrator | `modular_model_orchestrator` |
| LLM client | `openai.AzureOpenAI` chat completions, one user message |
| API version | `2024-02-01` |
| Deployment | `AZURE_OPENAI_DEPLOYMENT`, default `gpt-5-mini` |
| Endpoint | `AZURE_OPENAI_ENDPOINT` |
| Temperature | omitted; this deployment accepts only the default |
| Reasoning | `reasoning_effort="none"` |
| Single-slice output cap | `max_completion_tokens=110` |
| Tournament output cap | uncapped; `run_tournament_shootout.py` records the billed total |
| Response format | `json_object` |
| Slice block | 4 by 4 for regressors. Autoencoders use one 8 by 4 block, so an 8×2 encoder and a 2×8 decoder each stay whole. |
| History in the prompt | last 3 trials |
| Journal length | 5 committed scores |
| Eval samples | 50 |
| Divergence cutoff | loss `> 100` |
| Prompt integer range | the arm `int_bound`, default `-25` to `25` |
| Engine integer range | `-99` to `99`; the parser clamps a code onto that edge before scoring |
| Gamma after a divergent or rejected step | `0.015` to `0.035` |
| Gamma while improving | `0.030` to `0.065` |
| Storage | Azurite via `UseDevelopmentStorage=true` |

`local.settings.json` holds `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, and `BLOB_STORAGE_CONNECTION`. That file is local-only. Do not commit the key.

There is still an older two-layer path, `optimize_network_orchestrator`, which sends both layer tokens in one prompt. New single-candidate runs use `optimize_modular_orchestrator`. Runs that should compare three hypotheses use `modular_model_orchestrator`.

## How the token budget stays low

The model never sees a dense matrix. A layer that is wider than 4 on either side is cut into 4 by 4 blocks. A token for one block is `O(rows + cols)` integers, which for a full block is 4 + 4 codes plus a short gamma and up to 4 bias numbers.

The single-slice completion is a raw chat call:

- No system message.
- No tool schema and no agent-framework envelope.
- `reasoning_effort="none"` so the deployment does not bill a hidden reasoning trace.
- `max_completion_tokens=110` so the reply cannot grow past the output budget.
- History is three short lines. Status words are `div`, `rej`, and `ok`.
- Gamma is printed with trailing zeros removed, and biases are printed to two decimals, so the slice text does not carry full `double` expansions such as `0.20000000000000001`.

A measured check of that short prompt billed:

| Network | Slices | Prompt tokens | Completion tokens |
| --- | --- | --- | --- |
| 2-4-1 | 2 | 143 | 110 |
| 4-8-4-1 | 5 | 151 | 110 |
| 8-16-1 | 12 | 151 | 110 |

Prompt tokens stay under 160. Completions stop at the 110 cap (`finish_reason=length`). The host log prints `[tokens] prompt=... completion=... total=...` on every trial.

The short prompt for one slice looks like this:

```text
Slice stem@r0c0 4x4
- T1 12.0000 div
- T2 4.0000 rej
- T3 1.5000 ok
L 1.6997
w:r:1|g:0.0033|A:-99 -71 -42 -14|B:65 65 65 65
b:0.1 0.1 0.1 0.1
div g .015-.035 else .03-.065 ints -25..25
json {"strategy":"recovery","w":"r:1|g:.02|A:..|B:..","b":".."}
```

The word `json` is required by this API when `response_format` is `json_object`. The reply is a flat object with `strategy`, `w`, and `b`. If a field is missing or the JSON is cut off, that field stays on the slice that was sent in.

## Expanded tournament budget

The sub-250 ceiling applies to the single-slice loop above. `modular_model_orchestrator` does not set `max_completion_tokens`. The prompt stays small because it sends one slice token, a three-line history, a gradient hint, and a short role for each candidate. `scripts/verify_modular_sweep.py` prints the tokens the deployment billed and does not reject a run for its size.

A live check on this deployment, with that prompt and no completion cap, billed:

| Model | Prompt tokens | Completion tokens | Total | Integer changes (c1, c2, c3) | Winner |
| --- | --- | --- | --- | --- | --- |
| MlpRegressor 4-8-4-1 | 499 | 277 | 776 | 7, 8, 8 | c1, loss 1.6950 |
| BottleneckAutoencoder 8-2-8 | 485 | 238 | 723 | 6, 6, 6 | divergent, B codes above 99 |

The tournament completion has no token cap. The prompt tells the model to change at least two integers in `A` and `B` and never copy the active `w` string. The reply is a JSON object with `c1`, `c2`, and `c3`. `c1` is a small step, `c2` follows the gradient hint, and `c3` explores a wider gamma. If a candidate is missing, copies the active slice, or leaves fewer than two integer codes changed, that completion is discarded and the activity tries again, up to three times. Each run writes `reports/latest_report.md`, a timestamped copy next to it, and appends the raw reply to `reports/tournament_completions.jsonl`. After three misses the trial is recorded as `mutation_failed` and the checkpoint is not written. C++ also marks a candidate `unchanged` when its decoded block matches the active token, so a copied slice cannot win on rounding. C++ scores the remaining slices and commits the winner only when its loss is lower than the stored loss by more than `1e-4`. A code outside `-99` to `99` is clamped onto that edge before it is scored, so `100` does not throw. A divergent candidate is skipped. If none of the finite scores beat the stored loss by more than `1e-4`, the checksum stays the same.

## Layout

```text
orchestrators/          Durable training loop and the tournament evaluator
scripts/                verify_modular_sweep.py and local seed helpers
reports/                latest_report.md, timestamped copies, tournament_completions.jsonl
native/include/stevejob/
  core/                 config, hash, binary helpers
  models/               NeuralNetwork, ModelTemplate, MlpRegressor, BottleneckAutoencoder
  messaging/            TOTR tokenizer, encoding schemes, and SJC1 container
  compiler/             PromptCompiler
  storage/              Azurite client and sliding journal
  bridge/               ModularOrchestrator, EngineGateway, and the older two-layer pipeline
native/bindings/        pybind11 module
build/native/           tensor_engine.pyd
tools/                  Host registration and the thin Python wrappers
function_app.py         Functions host entry point
```

## Build the native engine

Requirements: Python 3.11+, CMake 3.18+, pybind11, and the Visual Studio 2022 C++ toolset on Windows. This repo uses Python 3.14 in `.venv`.

From the repository root:

```powershell
python -m pip install -r requirements.txt
cmake -S native -B native/build -G "Visual Studio 17 2022" -A x64 `
  -DPython3_EXECUTABLE="$PWD\.venv\Scripts\python.exe" `
  -Dpybind11_DIR="$PWD\.venv\Lib\site-packages\pybind11\share\cmake\pybind11"
cmake --build native/build --config Release --target tensor_engine
```

That writes `build/native/tensor_engine*.pyd`. Python loads it through `tools/__init__.py`. CMake's intermediate files stay in `native/build/`.

On Linux or macOS, use the default generator instead of Visual Studio:

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(pwd)/.venv/bin/python" \
  -Dpybind11_DIR="$(pwd)/.venv/lib/python3.11/site-packages/pybind11/share/cmake/pybind11"
cmake --build native/build
```

Adjust the `pybind11_DIR` path if your environment's site-packages location differs. The Azurite client is the WinHTTP build. The POSIX branch of that client throws.

## Run the tests

```powershell
python -m unittest discover tests -v
```

Storage tests talk to local Azurite on `127.0.0.1:10000`. `local.settings.json` should set `BLOB_STORAGE_CONNECTION` and `AzureWebJobsStorage` to `UseDevelopmentStorage=true`. You do not need a cloud storage account for local runs.

`scripts/verify_modular_sweep.py` seeds `MlpRegressor` and `BottleneckAutoencoder`, calls the deployment once each, and checks that the tournament total lands in 350–500 tokens. It reads the key from `local.settings.json` and does not print it.

## Run the Functions host

```powershell
$env:PYTHONPATH = "C:\Users\Cutie Magic 500\stevejob\stevejob\.venv\Lib\site-packages"
func start
```

Training route:

```text
POST http://localhost:7071/api/orchestrators/optimize_modular_orchestrator
POST http://localhost:7071/api/orchestrators/modular_model_orchestrator
```

The training loop is the durable orchestrator `modular_model_orchestrator`. The prompt for each slice is compiled in C++.

Stop the host when the run finishes. The listener is the process bound to port 7071. Azurite on port 10000 is a different process and should keep running.
