# Forward Analysis Distiller (`offline_models/forward_analysis`)

The **Forward Analysis Distiller** is a C++17 neural translation and numerical reasoning engine that ingests raw neural network forward-pass telemetry—multi-dimensional inputs, hidden layer activations, predictions, targets, and residuals—and distills them into:
1. **Calibrated Multi-Domain Numerical Predictions** across six mathematical analysis domains (`Statistical`, `Spectral`, `Geometric`, `Activation`, `Functional`, `Structural`).
2. **Factual, Token-Bounded Qualitative Diagnostic Text** across four structured observation fields (`pathway_activity`, `feature_sensitivity`, `representational_structure`, `functional_mapping`).

---

## Core Architecture Overview

```
Raw Forward Telemetry (Inputs, Layer Activations, Predictions, Targets, Residuals)
                                      │
                                      ▼
                     FeatureExtractorRegistry (6 Domains)
      Statistical │ Spectral │ Geometric │ Activation │ Functional │ Structural
                                      │
                                      ▼
                MultiFamilyTokenExtractor & ChannelSequenceFormatter
      • Channel Tokens      (C x 38)  ──► LinearInputAdapter + FamilyEmb[0]
      • Sample Tokens       (S x 16)  ──► LinearInputAdapter + FamilyEmb[1] (x2.0)
      • Relationship Tokens (12 x 16) ──► LinearInputAdapter + FamilyEmb[2] (x3.0)
      • Layer Tokens        (2 x 16)  ──► LinearInputAdapter + FamilyEmb[3] (x4.0, masked)
      • Global Tokens       (6 x 16)  ──► LinearInputAdapter + FamilyEmb[4] (x6.0)
                                      │
                                      ▼
                    Combined Sequence X_seq (T_total x 256)
                                      │
                                      ▼
                             PerceiverIOEncoder
           M = 8 learned latent queries cross-attend over T_total tokens
                                      │
                                      ▼
                        Latent Bottleneck Z (8 x 256)
                                      │
                                      ▼
                            EncodedRepresentation
       Global Mean Pool z_bar (256) + Domain-Routed Vectors z_d (0.5*z_bar + 0.5*Z[d])
                     ┌────────────────┴────────────────┐
                     │                                 │
                     ▼                                 ▼
       [AuxiliaryAlignmentHeads]         [AutoregressiveCausalDecoder]
   NumericalPredictionHeadRegistry:       • Cross-attends over all 8 rows of Z
   1. ActivationSparsityHead  (dim 8)     • Injects W_aux * d_cond (14-dim domain
   2. NumericalRankHead       (dim 4)       conditioning vector from heads)
   3. NarrowRepresentationHead(dim 4)     • Applies DecoderDomainGuidance logit
   4. LayerStructureHead      (dim 6)       calibration (layer count, rank regime)
   5. OutputAlignmentHead     (dim 9)     • Emits delexicalized slot tokens
   6. FeatureAssociationHead  (dim 5)       (<COUNT:*>, <DIM:*>, <COORD:*>)
   7. EigenSignPredictionHead (dim 4)     • Relexicalizes from ShapeSlotLexicon
   8. SpectralProfileHead     (dim 4)     • Verifies via ObservationConsistency
   + ObservationRegressionHead(dim 14)      with MeasuredFactsNarration fallback
```

---

## Directory Layout

* `include/forward_analysis/`
  * `domains/`: `AnalysisDomain` enum (`Statistical`, `Spectral`, `Geometric`, `Activation`, `Functional`, `Structural`) and `AnalysisDomainTraits`.
  * `features/`: `FeatureValueType`, `FeatureDescriptor`, `DomainFeatureExtractor`, `FeatureExtractorRegistry`, and the 6 concrete domain feature extractors plus `CustomProbeExtractor`.
  * `encoder/`: `ChannelFeatureExtractor`, `MultiFamilyTokenBatch`, `MultiFamilyTokenExtractor`, `LinearInputAdapter`, `ChannelSequenceFormatter`, and `PerceiverIOEncoder`.
  * `heads/`: `EncodedRepresentation`, `NumericalTargets`, `PredictionResult`, `LossResult`, `NumericalPredictionHead`, `DomainMlpHeadBase`, the 8 domain prediction heads, `ObservationRegressionHead`, `EigenSignHead`, `NumericalPredictionHeadRegistry`, and `AuxiliaryAlignmentHeads`.
  * `decoder/`: `AutoregressiveCausalDecoder`, `DecoderDomainGuidance`, `ObservationTokenizer`, `FieldDelexicalizer`, `BatchStatementDelexicalizer`, `ObservationDelexicalizer`, `ShapeSlotLexicon`, and OpenCL device mirror classes.
  * `description/`: `ObservationConsistency`, `BatchStatementChecker`, `ObservationNarrator`, `BatchStatisticsNarrator`, `MeasuredFactsNarration`, and `ObservationSimilarity`.
  * `bridge/`: `DualObjectiveDistillationBridge`, `BridgeCheckpoint`, and `DistillationResult`.
  * `compute/`: OpenCL runtime and double-precision GPU kernels (`OpenClRuntime`, `DeviceGemm`, `DecoderKernelSource`).
* `tests/test_forward_analysis.cpp`: Comprehensive unit test suite (3,304 assertions across all extractors, token families, heads, decoder, consistency verifier, and bridge pipeline).
* `experiments/forward_experiment.cpp`: CLI runner for training, evaluation, shuffled-text control, and the 5-stage feature ablation study (`Ablation1`–`Ablation5`).
* `demos/forward_pipeline_demo.cpp`: Self-contained demonstration of two-phase distillation training and inference.
* `docs/`: Architectural and mathematical documentation (`ARCH.md`, `DETAILS.md`, `MODULARITY.md`, `training_notes/1.md`, `scratch/desc0.txt`).

---

## Building and Running

```powershell
cmake -S offline_models/forward_analysis -B offline_models/forward_analysis/build -DCMAKE_BUILD_TYPE=Release
cmake --build offline_models/forward_analysis/build --config Release
```

### Unit Tests

```powershell
.\offline_models\forward_analysis\build\test_forward_analysis.exe
```

### Training & Evaluation CLI (`forward_experiment`)

```powershell
# Full refactored architecture (Level 5: Complete Architecture)
.\offline_models\forward_analysis\build\forward_experiment.exe `
  --name=Trial_G `
  --train=offline_models/forward_analysis/data/llm_train.jsonl `
  --select=offline_models/forward_analysis/data/llm_validation.jsonl `
  --final=offline_models/forward_analysis/data/llm_eval_ood.jsonl `
  --pretrain=25 --finetune=12 --patience=5 --slots=batch --ablation=5
```
