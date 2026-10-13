# stevejob

Tiny neural workflows for Azure Functions. A model is a C++ `ModelTemplate`. It owns the layers, the loss, the telemetry, the journal fields, the probe, and the encodings. The durable loop asks the LLM to edit one slice per trial. The model scores the full network and writes the checkpoint only when the loss drops by more than `1e-4`.

Python passes blob names and the compiled prompt. Numeric work, HMAC signing, HTTP to Azurite, checkpoint parsing, and prompt compilation run in `build/native/tensor_engine`. The guide for adding a model is [docs/USAGE.md](docs/USAGE.md). The system map is [docs/PROJECT_BREAKDOWN.md](docs/PROJECT_BREAKDOWN.md).

## Models that ship

```python
from tools import tensor_engine

tensor_engine.MlpRegressor().seed_to_storage("mlp.json", "mlp_journal.json", 10)
tensor_engine.BottleneckAutoencoder().seed_to_storage("ae.json", "ae_journal.json", 10)
tensor_engine.DemoScaleModel().seed_to_storage("demo.json", "demo_journal.json", 5)
```

`MlpRegressor` is 4-8-4-1, ReLU hidden layers and a linear head, scored as sin/cos MSE. Seed loss is about 1.6997. Its journal task is `regression`. Telemetry is `sparsity`, `variance`, and `conditioning`. The probe is comma-separated marks `+2`, `+1`, `0`, `-1`, `-2`.

`BottleneckAutoencoder` is 8-2-8, a ReLU encoder and a linear decoder, scored as reconstruction MSE. Seed loss is about 3.2818. Its journal task is `reconstruction`. Roster is `encode`, `decode`.

`DemoScaleModel` is 3-2-1, a tanh mix and a linear head. Its candidate slots are `hold` and `shift`. It lives in `tests/demo_scale_model.hpp` and is the small example next to `tests/test_model_modularity.py`.

`seed_to_storage` scores the network, packs an SJC1 blob in the `models` container, and writes the model's seed journal row under `evals`. The receipt contains `status`, `loss`, `checksum`, `architecture`, `layers`, and `directives`. It does not contain weight matrices.

`NeuralNetwork(...).seed_to_storage` still works. `ArchitectureCatalog` adopts that blob as `MlpRegressor` unless `architecture` is `bottleneck_ae` or `demo_scale`. A new loss belongs on a new `ModelTemplate` subclass registered in the catalog. See [docs/USAGE.md](docs/USAGE.md).

Activations are `LinearActivation`, `ReLUActivation`, `TanhActivation`, and `SigmoidActivation`. Default weights are `0.02 * (row - col)`. Default biases are `0.1`.

A shape list still builds a ReLU stack with a linear head and scores it as the regressor:

```python
tensor_engine.seed_stack_native([[4, 8], [8, 4], [4, 1]], "deep.json", "deep_journal.json")
```

## Tournament loop

`modular_model_orchestrator` is the training loop. `optimize_network_orchestrator` and `optimize_modular_orchestrator` use the same slice loop and the same C++ prompt compiler. One HTTP POST runs the trials. Each trial compiles a prompt, completes one user message, and asks `commit_tournament` to score every candidate slot.

The prompt sections, in order, are task, journal, telemetry, gradient probe, parameter encodings, candidate roles, and the response schema. The model fills each section. Weight tokens for the regressor and the autoencoder are rank-1:

```text
r:1|g:<gamma>|A:<row ints>|B:<col ints>
```

`W[row][col] = (A[row] * gamma) * (B[col] * gamma)`. `TotrRank1Scheme` clamps a code onto [-99, 99] before scoring. `constraint_line` prints `int_bound` (default -25 to 25). Gamma bands are the sentences in `candidate_roles`.

The commit gate:

| Result | Status | Blob |
| --- | --- | --- |
| Loss above 100, or the token cannot be decoded | `rejected_divergent` | unchanged |
| Fewer than two integer changes, and the probe is not a flat bias-only step | `rejected_zero_mutation` | unchanged, journal not appended |
| Loss drop of `1e-4` or less | `rejected` | unchanged, journal appended |
| Loss lower by more than `1e-4` | `committed_natively` | replaced, journal appended |

Start the host, then post a run. Set `PYTHONPATH` to the virtualenv site-packages so the worker can import the packages:

```powershell
$env:PYTHONPATH = "C:\Users\Cutie Magic 500\stevejob\stevejob\.venv\Lib\site-packages"
func start
```

```powershell
Invoke-RestMethod -Uri "http://localhost:7071/api/orchestrators/modular_model_orchestrator" `
  -Method Post `
  -ContentType "application/json" `
  -Body (@{
    max_trials = 10
    target_loss = 0.01
    checkpoint_blob = "mlp.json"
    journal_blob = "mlp_journal.json"
    journal_capacity = 10
    rolling_config = @{ visible_history_count = 3 }
    training_params = @{ int_bound = 25 }
  } | ConvertTo-Json)
```

Poll `statusQueryGetUri` from the 202 response until `runtimeStatus` is `Completed` or `Failed`. The response body includes a management key. Do not log that query string.

History rows include `trial`, `layer`, `loss`, `delta`, `status`, `winner`, and `winner_step`. `best_loss` updates only on `committed_natively`.

`scripts/run_tournament_shootout.py` runs ten trials each for `mlp_regressor` and `bottleneck_ae` and writes the scoreboard under `reports/`.

## Current configuration

| Setting | Value |
| --- | --- |
| Tournament orchestrator | `modular_model_orchestrator` |
| Other slice loops | `optimize_network_orchestrator`, `optimize_modular_orchestrator` |
| LLM client | `openai.AzureOpenAI` chat completions, one user message |
| API version | `2024-02-01` |
| Deployment | `AZURE_OPENAI_DEPLOYMENT`, default `gpt-5-mini` |
| Endpoint | `AZURE_OPENAI_ENDPOINT` |
| Temperature | omitted; this deployment accepts only the default |
| Reasoning | `reasoning_effort="none"` |
| Completion cap | none |
| Response format | `json_object` |
| Slice block | 4×4 on the regressor. The autoencoder uses an 8×4 envelope so encode and decode stay whole slices. |
| History in the prompt | last 3 journal rows, from `visible_history_count` |
| Journal length | the model's `journal_config().capacity`, or `journal_capacity` on the request |
| Eval samples | 50 on the regressor and the autoencoder. `DemoScaleModel` uses 4. |
| Divergence cutoff | loss `> 100` |
| Prompt integer range | `int_bound`, default `-25` to `25`, printed by the weight scheme |
| Engine integer range | `-99` to `99` on TOTR decode |
| Storage | Azurite via `UseDevelopmentStorage=true` |

`local.settings.json` holds `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, and `BLOB_STORAGE_CONNECTION`. That file is local-only. Do not commit the key.

## Layout

```text
docs/                   PROJECT_BREAKDOWN.md and USAGE.md
orchestrators/          Durable loops and the tournament extractor
scripts/                run_tournament_shootout.py, verify_modular_sweep.py, seed helpers
reports/                latest_report.md, timestamped copies, tournament_completions.jsonl
native/include/stevejob/
  models/               ModelTemplate, NeuralNetwork, catalog, MlpRegressor, BottleneckAutoencoder
  messaging/            TOTR, encoding schemes, output scheme interface, SJC1
  compiler/             PromptCompiler and CandidateReader
  storage/              Azurite client, metric fields, sliding journal
  bridge/               EngineGateway and Tournament
native/bindings/        pybind11 module
build/native/           tensor_engine.pyd
tests/                  demo_scale_model.hpp and the unit tests
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

`scripts/verify_modular_sweep.py` seeds `MlpRegressor` and `BottleneckAutoencoder`, calls the deployment once each, and prints the billed token counts. It reads the key from `local.settings.json` and does not print it.

## Run the Functions host

```powershell
$env:PYTHONPATH = "C:\Users\Cutie Magic 500\stevejob\stevejob\.venv\Lib\site-packages"
func start
```

Training routes:

```text
POST http://localhost:7071/api/orchestrators/modular_model_orchestrator
POST http://localhost:7071/api/orchestrators/optimize_modular_orchestrator
POST http://localhost:7071/api/orchestrators/optimize_network_orchestrator
```

The prompt for each slice is compiled in C++ from the model stored in the checkpoint.

Stop the host when the run finishes. The listener is the process bound to port 7071. Azurite on port 10000 is a different process and should keep running.
