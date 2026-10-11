"""Tournament parsing rejects a copied slice and accepts mutated A/B tokens."""

from __future__ import annotations

import unittest

from orchestrators.dynamic_injector import (
    TournamentParseError,
    build_parameterized_slice_prompt,
    collect_tournament,
    integer_distance,
    parse_tournament,
)

ACTIVE = "r:1|g:0.002|A:-99 -33 33 99|B:74 74 74 74"
BIAS = "0.1 0.1 0.1 0.1"
PAYLOAD = {
    "active_w": ACTIVE,
    "active_b": BIAS,
    "layer_name": "layer1@r0c0",
    "in_dim": 4,
    "out_dim": 4,
    "current_loss": 1.6997,
    "gradient_hint": "A:[+1, +1, +1, -1] B:[+1, +1, +1, +1]",
    "directives": {"task": "4-in regression.", "model_type": "MlpRegressor", "rules": "ints -25..25."},
    "training_params": {"int_bound": 25, "gamma_min": 0.02, "gamma_max": 0.07, "arm_name": "MLP-Standard"},
    "telemetry": {"condition": 2.7, "sparsity": 0.54, "variance": 0.004},
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
    def test_prompt_requires_a_real_mutation(self) -> None:
        prompt = build_parameterized_slice_prompt(PAYLOAD)
        self.assertIn("mutate at least 2 ints", prompt)
        self.assertIn("Never copy active w", prompt)
        self.assertIn("gamma <= 0.015", build_parameterized_slice_prompt({
            **PAYLOAD,
            "history": [{"trial": 1, "loss": 101.0, "delta": 0.0, "status": "rejected_divergent", "winner": "none"}],
        }))
        self.assertIn("0.01-0.03", build_parameterized_slice_prompt({
            **PAYLOAD,
            "history": [{"trial": 2, "loss": 1.2, "delta": -0.4, "status": "committed_natively", "winner": "c3"}],
        }))
        self.assertIn("JSON", prompt)
        self.assertIn("reasoning", prompt)
        self.assertIn(ACTIVE, prompt)
        self.assertIn("inside [-25, 25]", prompt)
        self.assertIn("'+2'", prompt)
        self.assertIn("'-1'", prompt)
        tight = build_parameterized_slice_prompt({
            **PAYLOAD,
            "training_params": {**PAYLOAD["training_params"], "int_bound": 15},
        })
        self.assertIn("inside [-15, 15]", tight)

    def test_standard_object_is_three_mutations(self) -> None:
        parsed, meta = parse_tournament(_object(), PAYLOAD)
        self.assertIn("Plus signs", meta["reasoning"])
        self.assertEqual(meta["confidence"], 0.72)
        self.assertEqual(meta["gammas"], {"c1": "0.025", "c2": "0.04", "c3": "0.06"})
        self.assertEqual([row["name"] for row in parsed], ["c1", "c2", "c3"])
        for row in parsed:
            self.assertGreaterEqual(integer_distance(row["w"], ACTIVE), 2)
            self.assertNotEqual(row["w"], ACTIVE)

    def test_factor_object_and_candidate_list_are_accepted(self) -> None:
        text = """{"candidates":[
            {"name":"c1","g":0.025,"A":[-10,-8,4,12],"B":[5,6,7,8],"b":[0.1,0.1,0.1,0.1]},
            {"name":"c2","w":{"gamma":0.04,"A":[-8,-6,6,14],"B":[8,9,10,11]},"bias":"0.1 0.1 0.1 0.1"},
            {"id":"c3","w":"r:1|g:0.06|A:2 4 6 8|B:-4 -3 -2 -1","b":"0.1 0.1 0.1 0.1"}
        ]}"""
        parsed, meta = parse_tournament(text, PAYLOAD)
        self.assertEqual(meta["gammas"]["c1"], "0.025")
        self.assertEqual(integer_distance(parsed[0]["w"], ACTIVE), 8)
        self.assertIn("|A:-8 -6 6 14|", parsed[1]["w"])
        self.assertEqual(parsed[2]["b"], BIAS)

    def test_copied_active_slice_is_rejected(self) -> None:
        active = "r:1|g:0.02|A:1 2 3 4|B:5 6 7 8"
        payload = {**PAYLOAD, "active_w": active}
        text = (
            '{"c1":{"w":"%s","b":"%s"},"c2":{"w":"%s","b":"%s"},"c3":{"w":"%s","b":"%s"}}'
            % (active, BIAS, active, BIAS, active, BIAS)
        )
        with self.assertRaises(TournamentParseError):
            parse_tournament(text, payload)

    def test_dead_probe_accepts_a_bias_raise_without_new_integers(self) -> None:
        active = "r:1|g:0.02|A:1 2 3 4|B:5 6 7 8"
        payload = {**PAYLOAD, "active_w": active, "gradient_hint": "A:[0, 0, 0, 0] B:[0, 0, 0, 0]"}
        text = (
            '{"reasoning":"All zeros, raise bias.","confidence":0.4,'
            '"c1":{"w":"%s","b":"0.4 0.4 0.4 0.4"},'
            '"c2":{"w":"%s","b":"0.5 0.5 0.5 0.5"},'
            '"c3":{"w":"%s","b":"0.6 0.6 0.6 0.6"}}'
            % (active, active, active)
        )
        parsed, meta = parse_tournament(text, payload)
        self.assertEqual(meta["reasoning"], "All zeros, raise bias.")
        self.assertEqual(integer_distance(parsed[0]["w"], active), 0)
        self.assertEqual(parsed[2]["b"], "0.6 0.6 0.6 0.6")

    def test_out_of_range_code_is_clamped_before_scoring(self) -> None:
        wide = _token("-101 -29 37 95", "100 72 70 102", "0.03")
        text = _object().replace(_token("-8 -6 6 14", "8 9 10 11", "0.04"), wide)
        parsed, _meta = parse_tournament(text, PAYLOAD)
        self.assertIn("A:-99 -29 37 95", parsed[1]["w"])
        self.assertIn("B:99 72 70 99", parsed[1]["w"])
        self.assertNotIn("-101", parsed[1]["w"])
        self.assertNotIn("100", parsed[1]["w"])
        listed = """{"candidates":[
            {"name":"c1","g":0.025,"A":[-10,-8,4,12],"B":[5,6,7,8],"b":"0.1 0.1 0.1 0.1"},
            {"name":"c2","g":0.04,"A":[-101,100,6,14],"B":[102,9,10,-100],"b":"0.1 0.1 0.1 0.1"},
            {"name":"c3","w":"r:1|g:0.06|A:2 4 6 8|B:-4 -3 -2 -1","b":"0.1 0.1 0.1 0.1"}
        ]}"""
        listed_rows, _listed_meta = parse_tournament(listed, PAYLOAD)
        self.assertIn("A:-99 99 6 14", listed_rows[1]["w"])
        self.assertIn("B:99 9 10 -99", listed_rows[1]["w"])

    def test_one_integer_change_is_rejected(self) -> None:
        near = "r:1|g:0.02|A:1 2 3 4|B:5 6 7 9"
        active = "r:1|g:0.02|A:1 2 3 4|B:5 6 7 8"
        payload = {**PAYLOAD, "active_w": active}
        text = (
            '{"c1":{"w":"%s","b":"%s"},"c2":{"w":"%s","b":"%s"},"c3":{"w":"%s","b":"%s"}}'
            % (near, BIAS, _token("-8 -6 6 14", "8 9 10 11"), BIAS, _token("2 4 6 8", "-4 -3 -2 -1"), BIAS)
        )
        with self.assertRaises(TournamentParseError):
            parse_tournament(text, payload)

    def test_retry_then_accept(self) -> None:
        seen: list[str] = []

        def complete(prompt: str) -> tuple[str, int, int, int]:
            seen.append(prompt)
            if len(seen) == 1:
                return ('{"c1":{"w":"%s","b":"%s"}}' % (ACTIVE, BIAS), 10, 20, 30)
            return (_object(), 11, 40, 51)

        result = collect_tournament(PAYLOAD, complete)
        self.assertFalse(result["mutation_failed"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["total_tokens"], 81)
        self.assertEqual(len(result["candidates"]), 3)
        self.assertIn("Plus signs", result["reasoning"])
        self.assertIn("Retry:", seen[1])

    def test_three_copies_do_not_return_candidates(self) -> None:
        def complete(prompt: str) -> tuple[str, int, int, int]:
            return ('{"note":"same"}', 5, 5, 10)

        result = collect_tournament(PAYLOAD, complete)
        self.assertTrue(result["mutation_failed"])
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["total_tokens"], 30)


if __name__ == "__main__":
    unittest.main()
