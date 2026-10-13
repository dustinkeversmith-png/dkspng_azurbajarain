"""Candidate extraction stays in Python. The prompt and the mutation gate stay in C++."""

from __future__ import annotations

import unittest
import uuid

from orchestrators.tournament_evaluator import TournamentParseError, extract_tournament
from tools import tensor_engine

ACTIVE = "r:1|g:0.002|A:-99 -33 33 99|B:74 74 74 74"
BIAS = "0.1 0.1 0.1 0.1"
PAYLOAD = {
    "architecture": "mlp_regressor",
    "rows": 4,
    "cols": 4,
    "active": {"w": ACTIVE, "b": BIAS},
}


def _token(left: str, right: str, gamma: str = "0.03") -> str:
    return f"r:1|g:{gamma}|A:{left}|B:{right}"


def _object() -> str:
    c1 = _token("-10 -8 4 12", "5 6 7 8", "0.025")
    c2 = _token("-8 -6 6 14", "8 9 10 11", "0.04")
    c3 = _token("2 4 6 8", "-4 -3 -2 -1", "0.06")
    return (
        '{"reasoning":"Plus signs step down, the minus sign steps up.","confidence":0.72,'
        '"c1":{"w":"%s","b":"%s"},"c2":{"w":"%s","b":"%s"},"c3":{"w":"%s","b":"%s"}}'
        % (c1, BIAS, c2, BIAS, c3, BIAS)
    )


class TournamentParseTest(unittest.TestCase):
    def test_compiled_prompt_carries_schemes_and_bounds(self) -> None:
        blob = f"prompt_{uuid.uuid4().hex[:8]}.json"
        journal = f"prompt_{uuid.uuid4().hex[:8]}.json"
        seeded = tensor_engine.MlpRegressor().seed_to_storage(blob, journal, 10)
        layer = list(seeded["layers"])[0]
        prompt = tensor_engine.compile_prompt(blob, journal, layer, 3, 25, "")
        self.assertIn("=== TASK SPECIFICATION ===", prompt)
        self.assertIn("=== SYSTEM TELEMETRY & DIAGNOSTICS ===", prompt)
        self.assertIn("=== DIRECTIONAL GRADIENT PROBE ===", prompt)
        self.assertIn("=== PARAMETER ENCODINGS & ACTIVE STATE ===", prompt)
        self.assertIn("mutate at least 2 ints", prompt)
        self.assertIn("Never copy active w", prompt)
        self.assertIn("JSON", prompt)
        self.assertIn("reasoning", prompt)
        self.assertIn("totr_rank1", prompt)
        self.assertIn("bias_delta", prompt)
        self.assertIn("regression_output", prompt)
        self.assertIn("stay in [-25, 25]", prompt)
        self.assertIn("'+2'", prompt)
        self.assertIn("'-1'", prompt)
        self.assertIn("0.015-0.035", prompt)
        tight = tensor_engine.compile_prompt(blob, journal, layer, 3, 15, "fix the ints")
        self.assertIn("stay in [-15, 15]", tight)
        self.assertIn("CORRECTION REQUIRED", tight)

    def test_standard_object_keeps_three_tokens(self) -> None:
        parsed, meta = extract_tournament(_object(), PAYLOAD)
        self.assertIn("Plus signs", meta["reasoning"])
        self.assertEqual(meta["confidence"], 0.72)
        self.assertEqual(meta["gammas"], {"c1": "0.025", "c2": "0.04", "c3": "0.06"})
        self.assertEqual([row["name"] for row in parsed], ["c1", "c2", "c3"])

    def test_factor_object_is_packed_without_clamping(self) -> None:
        text = """{"candidates":[
            {"name":"c1","g":0.025,"A":[-10,-8,4,12],"B":[5,6,7,8],"b":[0.1,0.1,0.1,0.1]},
            {"name":"c2","g":0.04,"A":[-101,100,6,14],"B":[8,9,10,11],"b":"0.1 0.1 0.1 0.1"},
            {"name":"c3","w":"r:1|g:0.06|A:2 4 6 8|B:-4 -3 -2 -1","b":"0.1 0.1 0.1 0.1"}
        ]}"""
        parsed, meta = extract_tournament(text, PAYLOAD)
        self.assertEqual(meta["gammas"]["c1"], "0.025")
        self.assertIn("-101", parsed[1]["w"])
        self.assertIn("100", parsed[1]["w"])

    def test_missing_candidate_is_rejected(self) -> None:
        with self.assertRaises(TournamentParseError):
            extract_tournament('{"c1":{"w":"%s","b":"%s"}}' % (ACTIVE, BIAS), PAYLOAD)

    def test_scheme_slots_are_caller_defined(self) -> None:
        scheme = {
            "slots": ["p1", "p2", "p3"],
            "fields": [{"key": "w", "role": "weights"}, {"key": "b", "role": "bias"}],
            "groups": ["candidates"],
        }
        text = (
            '{"reasoning":"custom slots","confidence":0.4,'
            '"p1":{"w":"%s","b":"%s"},"p2":{"w":"%s","b":"%s"},"p3":{"w":"%s","b":"%s"}}'
            % (ACTIVE, BIAS, ACTIVE, BIAS, ACTIVE, BIAS)
        )
        parsed, meta = extract_tournament(text, {**PAYLOAD, "scheme": scheme})
        self.assertEqual([row["name"] for row in parsed], ["p1", "p2", "p3"])
        self.assertEqual(meta["reasoning"], "custom slots")


if __name__ == "__main__":
    unittest.main()
