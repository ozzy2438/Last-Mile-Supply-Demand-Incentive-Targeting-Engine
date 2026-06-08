from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the LaDe last-mile supply-demand incentive targeting engine."
    )
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory containing LaDe CSV files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for processed outputs.")
    parser.add_argument("--demo", action="store_true", help="Run with deterministic synthetic LaDe-style data.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_pipeline(raw_dir=args.raw_dir, output_dir=args.output_dir, demo=args.demo)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

