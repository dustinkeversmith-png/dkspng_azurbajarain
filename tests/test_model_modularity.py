"""A custom architecture lives in demo_scale_model.hpp and writes its prompt, checkpoint, and journal."""

from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path
from typing import Any

from tools import tensor_engine

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
PROMPT = ARTIFACTS / "demo_scale_prompt.txt"
CHECKPOINT = ARTIFACTS / "demo_scale_checkpoint.json"
JOURNAL = ARTIFACTS / "demo_scale_journal.json"


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        return value
    return str(value)


class ModelModularityTest(unittest.TestCase):
    def test_custom_model_compiles_and_saves_its_prompt(self) -> None:
        blob = f"demo_{uuid.uuid4().hex[:8]}.json"
        journal = f"demo_{uuid.uuid4().hex[:8]}.json"
        model = tensor_engine.DemoScaleModel()
        seeded = _plain(model.seed_to_storage(blob, journal, 5))
        self.assertEqual(seeded["architecture"], "demo_scale")
        self.assertEqual(seeded["origin"], "demo_scale")
        self.assertEqual(seeded["directives"]["model_type"], "DemoScaleModel")
        self.assertEqual(list(seeded["layers"]), ["mix", "head"])

        prompt = tensor_engine.compile_prompt(blob, journal, "mix", 3, 25, "")
        journal_text = tensor_engine.read_journal_text(journal)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        PROMPT.write_text(prompt, encoding="utf-8")
        CHECKPOINT.write_text(json.dumps(seeded, indent=2) + "\n", encoding="utf-8")
        JOURNAL.write_text(journal_text if journal_text.endswith("\n") else journal_text + "\n", encoding="utf-8")

        saved = PROMPT.read_text(encoding="utf-8")
        self.assertEqual(saved, prompt)
        self.assertIn("Model: demo_scale", saved)
        self.assertIn("Journal scheme: task scale.", saved)
        self.assertIn("origin=demo_scale", saved)
        self.assertNotIn("winner=seed", saved)
        self.assertIn("[scale_span]:", saved)
        self.assertIn("Encoding: span_decimal", saved)
        self.assertIn("span = highest weight - lowest weight.", saved)
        self.assertIn("[mix_sign]:", saved)
        self.assertIn("Encoding: mix_sign", saved)
        self.assertIn("+1 means hold can keep the mix.", saved)
        self.assertIn("Weight Scheme: totr_rank1", saved)
        self.assertIn("W[row][col] = (A[row] * gamma) * (B[col] * gamma).", saved)
        self.assertIn("scale_output", saved)
        self.assertNotIn("regression_output", saved)
        self.assertIn("Probe Data: mix:+1", saved)
        self.assertIn('"hold"', saved)
        self.assertIn('"shift"', saved)

        checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["origin"], "demo_scale")
        self.assertEqual(checkpoint["status"], "seeded")
        journal_saved = JOURNAL.read_text(encoding="utf-8")
        self.assertIn('"task": "scale"', journal_saved)
        self.assertIn('"name": "origin"', journal_saved)
        self.assertIn("demo_scale", journal_saved)
        self.assertNotIn("winner", journal_saved)

    def test_canonical_mlp_and_ae_models_and_per_learning_config(self) -> None:
        mlp = tensor_engine.MlpRegressor()
        ae = tensor_engine.BottleneckAutoencoder()
        demo = tensor_engine.DemoScaleModel()

        self.assertEqual(mlp.config.max_candidates, 3)
        self.assertEqual(mlp.config.max_input_cols, 4)
        self.assertEqual(ae.config.max_input_cols, 8)
        self.assertEqual(demo.config.max_candidates, 2)
        self.assertEqual(demo.config.max_input_cols, 3)

        custom_cfg = mlp.config
        custom_cfg.max_learning_rate = 0.025
        custom_cfg.int_bound = 15
        mlp.set_config(custom_cfg)
        self.assertAlmostEqual(mlp.config.max_learning_rate, 0.025)
        self.assertEqual(mlp.config.int_bound, 15)

        mlp_blob = f"mlp_{uuid.uuid4().hex[:8]}.json"
        mlp_journal = f"mlp_{uuid.uuid4().hex[:8]}.json"
        mlp_seeded = _plain(mlp.seed_to_storage(mlp_blob, mlp_journal, 5))
        self.assertEqual(mlp_seeded["architecture"], "mlp_regressor")
        self.assertEqual(mlp_seeded["origin"], "mlp_regressor")
        mlp_prompt = tensor_engine.compile_prompt(mlp_blob, mlp_journal, "layer1@r0c0", 3, 25, "")
        self.assertIn("Model: mlp_regressor", mlp_prompt)
        self.assertIn("regression_output", mlp_prompt)
        self.assertIn("bias_delta", mlp_prompt)

        ae_blob = f"ae_{uuid.uuid4().hex[:8]}.json"
        ae_journal = f"ae_{uuid.uuid4().hex[:8]}.json"
        ae_seeded = _plain(ae.seed_to_storage(ae_blob, ae_journal, 5))
        self.assertEqual(ae_seeded["architecture"], "bottleneck_ae")
        self.assertEqual(ae_seeded["origin"], "bottleneck_ae")
        ae_prompt = tensor_engine.compile_prompt(ae_blob, ae_journal, "encode", 3, 25, "")
        self.assertIn("Model: bottleneck_ae", ae_prompt)
        self.assertIn("reconstruction_output", ae_prompt)
        self.assertIn("latent_utilization", ae_prompt)

    def test_model_owned_parsing_and_layer_diagnostics(self) -> None:
        ckpt = {
            "architecture": "mlp_regressor",
            "layer1": {
                "weights": [[0.1, -0.2], [0.3, 0.4]],
                "biases": [0.05, -0.05],
            },
        }
        diag = _plain(tensor_engine.layer_diagnostics(ckpt))
        self.assertIn("layer1", diag)
        self.assertEqual(diag["layer1"]["regression_elements"], 4)

        damp = tensor_engine.parse_dampening({"dampening_text": '{"dampening": 0.5}'}, "mlp_regressor")
        self.assertAlmostEqual(damp, 0.5)

        grads = _plain(
            tensor_engine.parse_gradients(
                {
                    "gradients": {
                        "layer1": {
                            "weights": "r:1|g:0.1|A:2 -1|B:1 3",
                            "biases": "q:0.01|D:5 -5",
                        }
                    }
                },
                ckpt,
                "mlp_regressor",
            )
        )
        self.assertEqual(len(grads["layer1"]["weights"]), 2)
        self.assertEqual(len(grads["layer1"]["biases"]), 2)

    def test_forward_analyzed_model_appends_analysis_to_prompt_checkpoint_and_journal(self) -> None:
        blob = f"fa_{uuid.uuid4().hex[:8]}.json"
        journal = f"fa_{uuid.uuid4().hex[:8]}.json"
        model = tensor_engine.ForwardAnalyzedModel()
        seeded = _plain(model.seed_to_storage(blob, journal, 6))

        # 1. Verify checkpoint document contains forward_analysis results
        self.assertEqual(seeded["architecture"], "forward_analyzed_mlp")
        self.assertEqual(seeded["origin"], "forward_analyzed_mlp")
        self.assertIn("fa_dead_units", seeded)
        self.assertIn("fa_effective_rank", seeded)
        self.assertIn("fa_target_correlation", seeded)
        self.assertIn("fa_sign_agreement", seeded)
        self.assertIn("fa_summary", seeded)
        self.assertIn("fa_pathway_activity", seeded)
        self.assertIn("fa_feature_sensitivity", seeded)
        self.assertIn("fa_representational_structure", seeded)
        self.assertIn("fa_functional_mapping", seeded)

        # 2. Verify compiled prompt contains forward_analysis telemetry & journal history
        prompt = tensor_engine.compile_prompt(blob, journal, "layer1@r0c0", 3, 25, "")
        self.assertIn("Model: forward_analyzed_mlp", prompt)
        self.assertIn("[fa_dead_units]:", prompt)
        self.assertIn("[fa_effective_rank]:", prompt)
        self.assertIn("[fa_target_correlation]:", prompt)
        self.assertIn("[fa_pathway_activity]:", prompt)
        self.assertIn("[fa_feature_sensitivity]:", prompt)
        self.assertIn("[fa_representational_structure]:", prompt)
        self.assertIn("[fa_functional_mapping]:", prompt)
        self.assertIn("Encoding: forward_analysis_distiller", prompt)
        self.assertIn("fa_summary=", prompt)

        # 3. Verify journal JSON contains forward_analysis metrics and narrations
        journal_text = tensor_engine.read_journal_text(journal)
        self.assertIn('"task": "forward_analyzed_regression"', journal_text)
        self.assertIn('"name": "fa_dead_units"', journal_text)
        self.assertIn('"name": "fa_effective_rank"', journal_text)
        self.assertIn('"name": "fa_target_corr"', journal_text)
        self.assertIn('"name": "fa_summary"', journal_text)
        self.assertIn('"name": "fa_pathway"', journal_text)
        self.assertIn('"name": "fa_mapping"', journal_text)


if __name__ == "__main__":
    unittest.main()
