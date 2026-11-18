import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CARD_DIR = DATA_DIR / "cards"
GOLDEN_DIR = ROOT / "evals" / "golden"

# LiteLLM model strings, so you can swap providers with one env var.
# Week 11-12: route cheap steps to READER_MODEL_CHEAP and compare cost vs accuracy.
READER_MODEL = os.getenv("READER_MODEL", "anthropic/claude-sonnet-5-5")
READER_MODEL_CHEAP = os.getenv("READER_MODEL_CHEAP", "anthropic/claude-haiku-5-5")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Papers are long. Week 1 keeps it simple: send up to this many characters.
# Later (week 5+) replace with section-aware selection. Measure the difference!
MAX_PAPER_CHARS = int(os.getenv("MAX_PAPER_CHARS", "120000"))
MAX_EXTRACTION_RETRIES = 2
