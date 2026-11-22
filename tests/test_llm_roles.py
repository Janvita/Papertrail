"""LLM role wiring: every role resolves to the provider/model in src/llm_config.py.
No network calls (LLM.__init__ only builds a client)."""
import pytest

from src import llm as llm_module
from src.llm_config import PROVIDERS, ROLES


@pytest.fixture(autouse=True)
def fake_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    llm_module._instances.clear()
    yield
    llm_module._instances.clear()


@pytest.mark.parametrize("role", ["main", "filter", "topic"])
def test_role_resolves_to_config_provider_and_model(role):
    spec = ROLES[role]
    assert str(llm_module.get_llm(role)) == f"{spec['provider']}/{spec['model']}"


@pytest.mark.parametrize("role", list(ROLES))
def test_every_role_uses_a_known_provider(role):
    assert ROLES[role]["provider"] in PROVIDERS


def test_role_instances_are_cached():
    assert llm_module.get_llm("topic") is llm_module.get_llm("topic")
