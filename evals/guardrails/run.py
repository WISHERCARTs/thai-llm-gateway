"""Measure the guardrails on the labelled sets in this folder.

    python -m evals.guardrails.run            # print the report
    python -m evals.guardrails.run --check    # also fail if a number drops below evals/guardrails/baseline.json

PII is scored per value (kind and exact text must match). Injection is scored per message at the
gateway's blocking threshold. The sets were written by hand for this repo, so the numbers show
known strengths and gaps, not performance on real traffic.
"""

import argparse
import json
import sys
from pathlib import Path

from gateway.guardrails.injection import score_injection
from gateway.guardrails.pii import find_pii

HERE = Path(__file__).parent
THRESHOLD = 0.7


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def eval_pii() -> dict:
    tp = fp = fn = 0
    misses, false_alarms = [], []
    for case in load("pii.jsonl"):
        expected = {(e["kind"], e["value"]) for e in case["expected"]}
        found = {(f.kind, f.value) for f in find_pii(case["text"])}
        tp += len(expected & found)
        fp += len(found - expected)
        fn += len(expected - found)
        misses += [{"text": case["text"], "missed": list(x)} for x in expected - found]
        false_alarms += [{"text": case["text"], "found": list(x)} for x in found - expected]
    return {
        "cases": len(load("pii.jsonl")),
        "precision": round(tp / (tp + fp), 3) if tp + fp else 1.0,
        "recall": round(tp / (tp + fn), 3) if tp + fn else 1.0,
        "misses": misses,
        "false_alarms": false_alarms,
    }


def eval_injection() -> dict:
    cases = load("injection.jsonl")
    attacks = [c for c in cases if c["attack"]]
    benign = [c for c in cases if not c["attack"]]
    missed = [c["text"] for c in attacks if not score_injection(c["text"]).blocked(THRESHOLD)]
    false_blocks = [c["text"] for c in benign if score_injection(c["text"]).blocked(THRESHOLD)]
    return {
        "attacks": len(attacks),
        "benign": len(benign),
        "detection_rate": round(1 - len(missed) / len(attacks), 3),
        "false_block_rate": round(len(false_blocks) / len(benign), 3),
        "missed": missed,
        "false_blocks": false_blocks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = {"pii": eval_pii(), "injection": eval_injection()}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.check:
        return 0
    base = json.loads((HERE / "baseline.json").read_text())
    failures = []
    for section, metric, higher_is_better in [
        ("pii", "precision", True),
        ("pii", "recall", True),
        ("injection", "detection_rate", True),
        ("injection", "false_block_rate", False),
    ]:
        now, then = report[section][metric], base[section][metric]
        if (now < then) if higher_is_better else (now > then):
            failures.append(f"{section}.{metric}: {now} (baseline {then})")
    for f in failures:
        print("REGRESSION", f, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
