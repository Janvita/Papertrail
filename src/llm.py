"""Provider-agnostic LLM wrapper with disk cache and retries.

Roles (main, filter, topic) are defined in llm_config.py. API keys come from .env.
"""
import hashlib, json, os, re, sys, time
from openai import OpenAI
from . import config, llm_config


class LLM:
    def __init__(self, provider, model):
        spec = llm_config.PROVIDERS[provider]
        env = spec["key_env"]
        key = "none" if env is None else os.environ.get(env)
        if not key:
            sys.exit(f"Set {env} in your .env file (provider: {provider})")
        self.provider, self.model = provider, model
        self.client = OpenAI(base_url=spec["base_url"], api_key=key)
        self.json_mode = spec["json_mode"]

    def __str__(self):
        return f"{self.provider}/{self.model}"

    def ask(self, prompt):
        key = hashlib.md5(f"{self}|{self.json_mode}|{prompt}".encode()).hexdigest()
        f = config.LLM_CACHE / f"{key}.txt"
        if f.exists():  # reruns cost nothing
            return f.read_text()

        kwargs = {"response_format": {"type": "json_object"}} if self.json_mode else {}
        for attempt in range(5):
            try:
                time.sleep(config.LLM_PAUSE)
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    **kwargs,
                )
                text = resp.choices[0].message.content
                f.write_text(text)
                return text
            except Exception as ex:
                msg = str(ex)
                low = msg.lower()
                if "per day" in low or "rpd" in low or "tpd" in low:
                    sys.exit("Daily quota used up. Rerun later (finished calls are cached).")
                if "429" in msg or "503" in msg or "rate" in low:
                    wait = 15 * (attempt + 1)
                    print(f"  LLM busy or rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("LLM unavailable after retries")

    def ask_json(self, prompt):
        text = self.ask(prompt + "\n\nReply with a single JSON object only, no prose, no code fences.")
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        return json.loads(text)


_instances = {}


def get_llm(role="main"):
    """role = 'main', 'filter' or 'topic'. Provider and model come from src/llm_config.py."""
    if role in _instances:
        return _instances[role]
    spec = llm_config.ROLES[role]
    _instances[role] = LLM(spec["provider"], spec["model"])
    return _instances[role]
