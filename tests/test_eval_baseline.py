"""The guardrail eval must not fall below its saved baseline (the same check CI runs)."""

import json
from pathlib import Path

from evals.guardrails.run import eval_injection, eval_pii

BASE = json.loads((Path(__file__).parents[1] / "evals/guardrails/baseline.json").read_text())


def test_pii_not_below_baseline():
    r = eval_pii()
    assert r["precision"] >= BASE["pii"]["precision"] and r["recall"] >= BASE["pii"]["recall"]


def test_injection_not_below_baseline():
    r = eval_injection()
    assert r["detection_rate"] >= BASE["injection"]["detection_rate"]
    assert r["false_block_rate"] <= BASE["injection"]["false_block_rate"]
