"""Loads configs/llm.yaml: the providers and the model used for each LLM role."""
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "llm.yaml"

with CONFIG_PATH.open() as f:
    _cfg = yaml.safe_load(f)

PROVIDERS = _cfg["providers"]
ROLES = _cfg["roles"]
