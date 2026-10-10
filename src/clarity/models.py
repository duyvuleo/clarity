"""Public model adapter; credentials are read from the user's environment only."""

from copy import deepcopy
import os


class Llm:
    @staticmethod
    def from_params(params):
        options = deepcopy(params)
        vendor = options.pop("vendor", "openai-compatible")
        if vendor == "codex":
            from clarity.codex_model import CodexChatModel

            allowed = {"model", "reasoning_effort", "timeout", "receipt_path"}
            if set(options) - allowed:
                raise ValueError(
                    "Codex options are model, reasoning_effort, timeout and receipt_path"
                )
            if "model" in options:
                options["model_name"] = options.pop("model")
            return CodexChatModel(**options)
        if vendor != "openai-compatible":
            raise ValueError("Use vendor 'openai-compatible' or 'codex'")
        from langchain_openai import ChatOpenAI

        if any(key in options for key in ["api_key", "password", "secret", "oci_credentials_path"]):
            raise ValueError(
                "Credentials must come from environment variables, not configuration files"
            )
        key_env = options.pop("api_key_env", "OPENAI_API_KEY")
        url_env = options.pop("base_url_env", "OPENAI_BASE_URL")
        model_env = options.pop("model_env", None)
        if model_env:
            model = os.environ.get(model_env)
            if not model:
                raise ValueError(f"Set {model_env} to the model name provided by your endpoint")
            options["model"] = model
        if not options.get("model"):
            raise ValueError("A model or model_env is required")
        key = os.environ.get(key_env)
        if not key:
            raise ValueError(f"Set {key_env} in your environment before running live generation")
        options["api_key"] = key
        if os.environ.get(url_env):
            options["base_url"] = os.environ[url_env]
        options.setdefault("timeout", 120)
        options.setdefault("max_retries", 2)
        return ChatOpenAI(**options)

    @staticmethod
    def get_model(model_name):
        return Llm.from_params({"model": model_name})
