from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from kimi_cli.app import KimiCLI
from kosong.message import Message

from kimi_agent_sdk import Session
from kimi_agent_sdk._session import ExportedContext


class _DummySession:
    def __init__(self, context_file: Path) -> None:
        self.context_file = context_file
        self.id = "test-session"


class _DummySoul:
    model_name = "dummy"
    status = cast(Any, None)


class _DummyCLI:
    def __init__(self, context_file: Path) -> None:
        self.session = _DummySession(context_file)
        self.soul = _DummySoul()

    async def run(self, *_args: Any, **_kwargs: Any) -> Any:
        return
        yield


@pytest.mark.asyncio
async def test_export_empty_context(tmp_path: Path) -> None:
    context_file = tmp_path / "context.jsonl"
    context_file.write_text("", encoding="utf-8")

    session = Session(cast(KimiCLI, _DummyCLI(context_file)))
    result = await session.export()

    assert isinstance(result, ExportedContext)
    assert result.system_prompt is None
    assert result.messages == []
    assert result.checkpoints == []
    assert result.usages == []


@pytest.mark.asyncio
async def test_export_mixed_records(tmp_path: Path) -> None:
    context_file = tmp_path / "context.jsonl"
    records = [
        {"role": "_system_prompt", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "_usage", "token_count": 42},
        {"role": "_checkpoint", "id": 0},
        {"role": "user", "content": "How are you?"},
    ]
    context_file.write_text(
        "".join(json.dumps(r) + "\n" for r in records),
        encoding="utf-8",
    )

    session = Session(cast(KimiCLI, _DummyCLI(context_file)))
    result = await session.export()

    assert isinstance(result, ExportedContext)
    assert result.system_prompt == "You are a helpful assistant."
    assert len(result.messages) == 3
    assert result.messages[0].role == "user"
    assert result.messages[0].extract_text() == "Hello"
    assert result.messages[1].role == "assistant"
    assert result.messages[1].extract_text() == "Hi there!"
    assert result.messages[2].role == "user"
    assert result.messages[2].extract_text() == "How are you?"
    assert result.usages == [42]
    assert result.checkpoints == [0]


@pytest.mark.asyncio
async def test_export_skips_invalid_lines(tmp_path: Path) -> None:
    context_file = tmp_path / "context.jsonl"
    context_file.write_text(
        '{"role": "_system_prompt", "content": "valid"}\n'
        'not json\n'
        '{"role": "_usage", "token_count": 10}\n'
        '{"invalid": "no role"}\n'
        '{"role": "_checkpoint", "id": 1}\n',
        encoding="utf-8",
    )

    session = Session(cast(KimiCLI, _DummyCLI(context_file)))
    result = await session.export()

    assert result.system_prompt == "valid"
    assert result.usages == [10]
    assert result.checkpoints == [1]
    assert result.messages == []
