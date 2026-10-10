import json
from types import SimpleNamespace

import pytest

from clarity.models import Llm


def test_codex_adapter_uses_structured_response_without_tools(monkeypatch, tmp_path):
    from clarity import codex_model

    monkeypatch.setattr(codex_model.shutil, "which", lambda name: "codex")
    observed = {}

    def run(arguments, **kwargs):
        observed.update(arguments=arguments, input=kwargs["input"])
        events = [
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": json.dumps({"response_json": json.dumps({"outcome": "valid"})}),
                },
            },
            {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}},
        ]
        return SimpleNamespace(returncode=0, stdout="\n".join(map(json.dumps, events)), stderr="")

    monkeypatch.setattr(codex_model.subprocess, "run", run)
    receipt = tmp_path / "receipts.jsonl"
    options = {"vendor": "codex", "receipt_path": str(receipt)}
    model = Llm.from_params(options)
    response = model.invoke("Return the requested JSON classification")
    assert json.loads(response.content) == {"outcome": "valid"}
    assert "--ephemeral" in observed["arguments"]
    assert observed["arguments"][observed["arguments"].index("--sandbox") + 1] == "read-only"
    assert "--ignore-user-config" in observed["arguments"]
    assert options == {"vendor": "codex", "receipt_path": str(receipt)}
    assert json.loads(receipt.read_text())["status"] == "success"


def test_codex_adapter_rejects_tool_use(monkeypatch):
    from clarity import codex_model

    monkeypatch.setattr(codex_model.shutil, "which", lambda name: "codex")
    event = {"type": "item.completed", "item": {"type": "command_execution"}}
    monkeypatch.setattr(
        codex_model.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(event), stderr=""),
    )
    with pytest.raises(ValueError, match="used tools"):
        Llm.from_params({"vendor": "codex"}).invoke("Return JSON")


def test_codex_configuration_rejects_credentials():
    with pytest.raises(ValueError, match="Codex options"):
        Llm.from_params({"vendor": "codex", "api_key": "fixture"})
