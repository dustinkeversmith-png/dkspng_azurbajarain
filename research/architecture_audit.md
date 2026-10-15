# Phase 0 — Architecture Audit: Forward Analysis, Network Scaling, Problem Complexity, and Encoding Rate

## 1. Executive Summary

This audit inspects the existing neural-network, storage, server, and forward-analysis implementations across `native/`, `online_models/`, `offline_models/forward_analysis/`, and `orchestrators/` prior to constructing and executing the scientific benchmark suite.

### Key Architectural Findings
1. **Two Coupled Neural Subsystems**:
   - **Online Target Models (`stevejob::models::ModelTemplate` & `NeuralNetwork`)**: Multi-layer feed-forward networks (`Linear`, `ReLU`, `Tanh`, `Sigmoid`) stored as `SJC1` binary checkpoints in Azurite (`models/` container) with sliding history journals in `evals/`. Models are optimized either via the Azure Durable Functions server (`modular_model_orchestrator` calling Azure OpenAI and C++ `EngineGateway::commit_tournament`) or via native C++ optimization steps.
   - **Offline Forward-Analysis Engine (`offline_models/forward_analysis`)**: A C++17 neural telemetry analyzer comprising a 6-domain `FeatureExtractorRegistry` (`Statistical`, `Spectral`, `Geometric`, `Activation`, `Functional`, `Structural`), 5-family `MultiFamilyTokenExtractor`, `PerceiverIOEncoder` ($M=8, D_{\text{model}}=256$), 8 domain-routed `NumericalPredictionHead` modules, an `AutoregressiveCausalDecoder`, and exact deterministic `ObservationExtractor` / `MeasuredFactsNarration`.
2. **Exact Semantics of Forward Analysis**:
   - Forward analysis does **not** alter the mathematical definition of the forward pass $\hat{Y} = f_\theta(X)$.
   - Instead, given a forward pass on a batch $X \in \mathbb{R}^{N \times D_{\text{in}}}$ producing hidden layer activations $\{H_\ell \in \mathbb{R}^{N \times W_\ell}\}_{\ell=1}^L$, predictions $\hat{Y} \in \mathbb{R}^{N \times D_{\text{out}}}$, targets $Y \in \mathbb{R}^{N \times D_{\text{out}}}$, and residuals $R = \hat{Y} - Y$, forward analysis computes:
     1. **Activation Pathway Diagnostics (`observe_layer`, `observe_gating`)**: Dead unit indices $\{u : \max_n |H_\ell(n, u)| \le 10^{-6}\}$, always-active units, nonpositive sample rows ($\max_u H_\ell(n, u) \le 0$), and half-space gating coordinates.
     2. **Spectral & Geometric Manifold Diagnostics (`observe_layer`, `eigen_signs`)**: Uncentered effective rank ($\sigma_i(H_\ell) \ge 0.05\,\sigma_0(H_\ell)$), centered numerical rank ($\sigma_i(\tilde{H}_\ell) > 10^{-3}$), narrow-representation indicator ($\text{rank} \le 2$), 1D ray collapse indicator ($\sigma_1 \le 10^{-3}\sigma_0$), and input zero-diagonal covariance eigenvalue signs.
     3. **Feature Sensitivity & Input Association (`observe_drivers`, `observe_input_association`)**: Principal-axis driver correlations ($r \ge 0.75$) and input-to-final-hidden Pearson associations ($a_c = \max_k |\text{corr}(X_{:,c}, H_{L,:,k})|$).
     4. **Functional Mapping Alignment (`observe_output`)**: Prediction-to-target Pearson correlation, range ratio $\frac{\max \hat{y} - \min \hat{y}}{\max y - \min y}$, L1 amplitude ratio, sign agreement fraction, order preservation, and flat-output collapse ($\text{std}(\hat{y}) \le 0.2\,\text{std}(y)$).
3. **How Forward Analysis Influences Training**:
   - **In the Azure Functions / LLM Tournament Loop (`ForwardAnalyzedModel`)**: Appends `fa_*` scalar and narrative tokens to the compiled slice prompt (`derive_training_telemetry`), checkpoint receipt (`checkpoint_document`), and sliding journal (`seed_record`, `trial_record`), and sets `candidate_roles` rules instructing candidates `c1/c2/c3` to revive dead units via biases, expand effective rank, and align output range/correlation.
   - **In Continuous / Analytical Parameter Updates**: Modulates parameter updates using the four measured `ObservationExtractor` mechanisms:
     - **M1 (`dead_revival`)**: Targeted pre-activation bias/weight revival on silent ReLU units ($u \in \text{dead\_units}$) whose analytical backprop gradient is identically zero.
     - **M2 (`rank_stabilization`)**: Spectral rank-expanding perturbation when hidden layers suffer rank collapse ($\text{effective\_rank} < \min(W_\ell, \text{target\_rank})$ or `narrow_representation == true`).
     - **M3 (`functional_calibration`)**: Adaptive head/step-size gain calibration from `range_ratio`, `sign_agreement_ratio`, and `flat` output detection.
     - **M4 (`driver_conditioning`)**: Input-coordinate gradient balancing from ` observe_input_association` (`strongest_inputs` vs. `weakest_inputs`).

---

## 2. Actual Meaning and Units of Encoding Rate

In this repository, encoding rate operates across two coupled interfaces:

1. **Input / Coordinate Feature Encoding Rate ($R_{\text{enc}} \in \{4, 8, 16, 32\}$ channels)**:
   - In `MlpRegressor` and `ForwardAnalyzedModel`, low-dimensional problem coordinates are projected into $D_{\text{in}}$ phase/frequency-encoded input channels ($\sin/\cos$ harmonics and phase shifts).
   - Across our 4 logarithmically spaced levels $E_1=4, E_2=8, E_3=16, E_4=32$, a problem coordinate vector $z \in [-1, 1]^{d_0}$ is mapped into $R_{\text{enc}}$ multiscale Fourier/phase features with frequencies $\omega_k = 2^{\lfloor k/2 \rfloor} \pi$, giving spectral bandwidth $\Omega_{\max} \propto 2^{R_{\text{enc}}/2}$.
2. **Slice / Token Quantization Encoding Rate (`TotrRank1Scheme` & `BiasDeltaScheme`)**:
   - Weight slices of shape $B_r \times B_c$ are encoded as `r:1|g:<gamma>|A:<row ints>|B:<col ints>` with integer code bound $Q = \text{int\_bound} \in \{12, 24, 48, 96\}$ ($\log_2(2Q+1) \in \{4.64, 5.61, 6.60, 7.60\}$ bits/code).

To separate **encoding resolution** from **network parameter count** (Group E), our benchmark supports both:
- **Standard Coupled Encoding**: $D_{\text{in}} = R_{\text{enc}}$, where hidden layer widths $(W_1, W_2)$ are solved so total trainable parameters $P = (D_{\text{in}}+1)W_1 + (W_1+1)W_2 + (W_2+1)D_{\text{out}}$ match the target parameter budget ($1\text{K}, 10\text{K}, 100\text{K}, 1\text{M}, 10\text{M}$).
- **Fixed-Dimension Capacity-Matched Encoding (Group E Control)**: Fixed $D_{\text{in}} = 32$ where only the first $R_{\text{enc}} \in \{4, 8, 16, 32\}$ frequency channels are active and remaining channels are masked to zero, keeping layer shapes and parameter count 100% identical across $E_1..E_4$.

---

## 3. Existing Limitations, Confounders, and Required Engineering Changes

1. **Confounder 1 — Deterministic Skew-Symmetric Weight Initialization (`LayerSpec::initialize_weights`)**:
   - Default initialization sets $W_{r,c} = 0.02(r - c)$ and $b_c = 0.1$. Because $r - c$ is a sum of two rank-1 matrices, every layer of any width $W \ge 3$ starts with **exact rank 2**, and there is no random seed dependence.
   - **Resolution**: Add seeded stochastic initialization (He/Xavier normal with deterministic `seed`) to `BenchmarkModelTemplate` while preserving `LayerSpec::initialize_weights` for existing production models.
2. **Confounder 2 — Azurite `NativeBlobClient` Hardcoded 16 KB / 1 MB Caps (`native_blob_client.hpp:40, 564`)**:
   - `NativeBlobClient` had `kMaxBlobBytes = 16384` (16 KB) and a 1 MB WinHTTP read buffer cap. While a 1K-parameter network is ~8 KB in `SJC1`, 10K (~80 KB), 100K (~800 KB), 1M (~8 MB), and 10M (~80 MB) exceed 16 KB.
   - **Resolution**: Allow `NativeBlobClient` to accept a configurable `max_blob_bytes` (defaulting to `16384` for backward compatibility, or raised up to `128 MB` when `ModelCheckpoint` / `BenchmarkModelTemplate` stores larger networks).
3. **Confounder 3 — `SJC1` Layer Name 64-Byte Limit (`sjc1_scheme.hpp:270`)**:
   - `SJC1Compressor::pack_body` stores `layer.name + \x1f + act + \x1f + 0 + \x1f + architecture` in the first layer's name and requires `name.size() <= 64`.
   - **Resolution**: Encode `BenchmarkModelTemplate` architecture strings compactly as `bm_<P>_<S>_<E>_<F>_<M>_<seed>` ($\le 36$ bytes) so `ArchitectureCatalog::visit` reconstructs the exact `BenchmarkModelTemplate` configuration when unpacked from Azurite by the Azure Functions server.
4. **Confounder 4 — $O(N W_\ell^2)$ SVD in `ObservationExtractor` for Wide Layers ($1\text{M}$–$10\text{M}$ Params)**:
   - For $1\text{M}$–$10\text{M}$ parameter networks, hidden widths reach $W_\ell \in [1000, 3160]$ while batch size $N \in [64, 256]$. Computing full SVD directly on $N \times W_\ell$ is $O(N W_\ell^2)$.
   - **Resolution**: When $W_\ell > 2N$, compute singular values via the $N \times N$ symmetric Gram matrix $G = H_\ell H_\ell^T$ ($\sigma_i(H_\ell) = \sqrt{\max(0, \lambda_i(G))}$), which is mathematically identical and $20\times$–$50\times$ faster.

---

## 4. Benchmark Problem Families (`P1`–`P6`) and Research Groups (`A`–`L`)

### Problem Families
- **P1 (`analytic`)**: Analytic function approximation (polynomial, sinusoidal, exponential, and mixed nonlinear interactions), evaluated on in-domain interpolation and out-of-domain extrapolation regions. Continuous difficulty parameter $\delta$: polynomial/nonlinearity order.
- **P2 (`multiscale`)**: Multiscale signals (low/high-frequency mixtures, chirps, sharp $\tanh$ transitions, nonstationary localized wavepackets). Continuous difficulty parameter $\delta$: high-to-low frequency ratio. Measures high-frequency spectral recovery error.
- **P3 (`compositional`)**: Compositional functions (3-level nested nonlinearities, arithmetic products, conditional half-space gating rules, and cross-coordinate interactions). Continuous difficulty parameter $\delta$: composition depth / gating sharpness.
- **P4 (`ode_hamiltonian`)**: Nonlinear Hamiltonian pendulum & coupled anharmonic oscillator dynamics ($H(q,p) = \frac{1}{2}p^2 + (1-\cos(\delta q)) + \frac{\kappa}{4}q^4$). Predicts multi-step state transition and energy; measures trajectory MSE, long-horizon rollout drift, and energy conservation violation.
- **P5 (`pde_burgers`)**: 1D Viscous Advection-Diffusion / Burgers PDE ($u_t + u u_x = \nu u_{xx}$). Predicts spatiotemporal field $(u, u_x, u_t)$; measures field MSE, interior PDE residual $\|u_t + u u_x - \nu u_{xx}\|_2^2$, and initial/boundary condition violation, with held-out temporal extrapolation ($t \in [0.75, 1.0]$).
- **P6 (`coupled_transport`)**: Multi-body / Coupled Nonlinear Damped-Driven Duffing-Lorenz Transport System. High-complexity nonlinear closure with continuous Reynolds/nonlinearity parameter $\delta$, measuring state accuracy, physical power-balance law residual, and OOD generalization.

### Experiment Groups (`A`–`L`)
- **Group A**: Baseline efficacy (`FA_DISABLED` vs. `FA_ENABLED`).
- **Group B**: Model scaling (`FA` $\times$ Parameter budgets `1K, 10K, 100K, 1M, 10M`).
- **Group C**: Problem complexity (`FA` $\times$ Problem families `P1..P6` and continuous difficulty $\delta$).
- **Group D**: Encoding scaling (`FA` $\times$ Encoding rates `E1=4, E2=8, E3=16, E4=32`).
- **Group E**: Encoding-capacity interaction (Parameters $\times$ Encoding, including fixed-parameter padded control vs. capacity-growing encoding).
- **Group F**: Full factorial interaction ($5 \times 6 \times 4 \times 2 \times 5 = 1,200$ paired runs).
- **Group G**: Data efficiency ($N_{\text{train}} \in \{128, 512, 2048, 8192, 32768, 65536\}$ samples).
- **Group H**: Noise robustness ($\sigma_{\text{noise}} \in \{0\%, 1\%, 2.5\%, 5\%, 10\%\}$ normalized target noise).
- **Group I**: Extrapolation (In-domain interpolation vs. out-of-domain spatial/temporal/amplitude extrapolation).
- **Group J**: Physics consistency (Supervised data-only loss vs. physics-informed PDE/Hamiltonian residual loss).
- **Group K**: Computational efficiency (Step-matched vs. wall-clock time-matched and FLOP-matched comparisons, plus live Azure Functions server token/latency comparisons).
- **Group L**: Mechanism ablations (`disabled`, `enabled_full`, `dead_only`, `rank_only`, `func_only`, `driver_only`, `sham_noise`).
