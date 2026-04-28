from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaroute.eval.metrics import compute_metrics
from adaroute.utils.io import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AdaRoute-MM JSONL results.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = compute_metrics(read_jsonl(args.input))
    write_json(args.output, summary)
    print(f"Saved summary to: {args.output}")


if __name__ == "__main__":
    main()
