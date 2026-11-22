"""Run: python -m src "your topic" [--max-papers N] [--min-relevance N] [--no-extract] [--out DIR]"""
import argparse
from .pipeline import run


def main():
    ap = argparse.ArgumentParser(description="Build a literature survey for a topic.")
    ap.add_argument("topic")
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument("--min-relevance", type=float, default=None)
    ap.add_argument("--no-extract", action="store_true", help="skip per-paper summaries")
    ap.add_argument("--out", default=None, help="output directory (default: survey_<topic-slug>)")
    args = ap.parse_args()

    run(args.topic, max_papers=args.max_papers, min_relevance=args.min_relevance,
        do_extract=not args.no_extract, out_dir=args.out)


if __name__ == "__main__":
    main()
