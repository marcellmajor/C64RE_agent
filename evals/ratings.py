"""Explicit CLI export of locally rated turns to a LangSmith dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.research_notebook import (
    rated_turn_examples,
    sync_ratings_to_langsmith,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        help="LangSmith dataset name. Omit with --local-only.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Print joined local rating examples without network writes.",
    )
    args = parser.parse_args()
    if args.local_only:
        print(json.dumps(rated_turn_examples(args.session_dir), indent=2))
        return
    if not args.dataset:
        parser.error("--dataset is required unless --local-only is used")
    result = sync_ratings_to_langsmith(
        args.session_dir, dataset_name=args.dataset,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
