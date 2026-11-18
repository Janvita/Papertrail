"""Usage:
    python -m papertrail.cli 2301.12345
    python -m papertrail.cli https://www.biorxiv.org/content/10.1101/XXXX.full.pdf
    python -m papertrail.cli data/pdfs/my_paper.pdf --model anthropic/claude-haiku-5-5
"""

import argparse
import json

from papertrail.config import READER_MODEL
from papertrail.graph import build_graph
from papertrail.llm import USAGE_LOG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--model", default=READER_MODEL)
    args = ap.parse_args()

    final = build_graph().invoke({"source": args.source, "model": args.model})
    if final.get("card"):
        print(final["card"].model_dump_json(indent=2))
    else:
        print("Extraction failed:", [e for e in final.get("errors", []) if not e.startswith("RAW::")])
    print("\nretries:", final.get("retries"), "| usage:", json.dumps(USAGE_LOG))


if __name__ == "__main__":
    main()
