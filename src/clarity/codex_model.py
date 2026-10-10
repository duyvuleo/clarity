"""Use authenticated Codex CLI inference as a LangChain chat model.

Only benchmark prompts are sent. CLI credentials are neither read by this adapter
nor copied into outputs. Each invocation is ephemeral and uses a read-only sandbox.
"""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

ENVELOPE_SCHEMA = {
    "type": "object",
    "properties": {"response_json": {"type": "string"}},
    "required": ["response_json"],
    "additionalProperties": False,
}
FORBIDDEN_ITEMS = {"command_execution", "mcp_tool_call", "web_search", "file_change"}


class CodexChatModel(BaseChatModel):
    model_name: str | None = None
    reasoning_effort: str = "low"
    timeout: float = 180
    receipt_path: str | None = None

    @property
    def _llm_type(self):
        return "codex-cli"

    @property
    def _identifying_params(self):
        return {
            "model": self.model_name or "CLI default",
            "reasoning_effort": self.reasoning_effort,
        }

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any):
        executable = shutil.which("codex")
        if not executable:
            raise ValueError("Install Codex CLI and sign in before using vendor 'codex'")
        if stop:
            raise ValueError("Codex CLI does not support explicit stop sequences in this adapter")
        payload = "\n\n".join(f"{message.type.upper()}:\n{message.content}" for message in messages)
        instruction = (
            "Act solely as a language model for the benchmark step below. Do not use tools, "
            "read local files, browse, edit files, or send messages. All necessary context is "
            "included. Follow the requested inner output schema. Return the JSON serialization "
            "of that answer in the response_json string of the outer response envelope.\n\n"
        )
        started = time.monotonic()
        receipt = {
            "provider": "codex-cli",
            "requested_model": self.model_name or "CLI default",
            "reasoning_effort": self.reasoning_effort,
            "prompt_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "prompt_characters": len(payload),
            "tool_items": 0,
        }
        try:
            with tempfile.TemporaryDirectory(prefix="clarity-codex-") as directory:
                schema = Path(directory) / "response-schema.json"
                schema.write_text(json.dumps(ENVELOPE_SCHEMA), encoding="utf-8")
                arguments = [
                    executable,
                    "exec",
                    "--ignore-user-config",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "--sandbox",
                    "read-only",
                    "--json",
                    "-c",
                    f'model_reasoning_effort="{self.reasoning_effort}"',
                    "-c",
                    "features.shell_tool=false",
                    "-c",
                    'web_search="disabled"',
                    "--output-schema",
                    str(schema),
                    "-",
                ]
                if self.model_name:
                    arguments[2:2] = ["--model", self.model_name]
                result = subprocess.run(
                    arguments,
                    input=instruction + payload,
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            events = []
            for line in result.stdout.splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            items = [
                event.get("item", {}) for event in events if event.get("type") == "item.completed"
            ]
            receipt["tool_items"] = sum(item.get("type") in FORBIDDEN_ITEMS for item in items)
            receipt["usage"] = next(
                (
                    event.get("usage", {})
                    for event in reversed(events)
                    if event.get("type") == "turn.completed"
                ),
                {},
            )
            if receipt["tool_items"]:
                raise ValueError("Codex used tools during a model-only benchmark request")
            if result.returncode:
                # CLI stderr can contain account or transport details; never copy it into artifacts.
                raise ValueError(
                    f"Codex CLI inference failed with exit code {result.returncode}; check login and model availability"
                )
            answers = [item["text"] for item in items if item.get("type") == "agent_message"]
            if not answers:
                raise ValueError("Codex CLI returned no final model response")
            envelope = json.loads(answers[-1])
            answer = envelope["response_json"]
            if not isinstance(answer, str) or not isinstance(json.loads(answer), dict):
                raise ValueError("Codex response_json must encode a JSON object")
            receipt["status"] = "success"
            receipt["response_sha256"] = hashlib.sha256(answer.encode()).hexdigest()
            receipt["response_characters"] = len(answer)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])
        except (subprocess.TimeoutExpired, OSError, ValueError, KeyError) as error:
            receipt["status"] = "failure"
            receipt["error_type"] = type(error).__name__
            raise
        finally:
            receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
            if self.receipt_path:
                path = Path(self.receipt_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(receipt) + "\n")
