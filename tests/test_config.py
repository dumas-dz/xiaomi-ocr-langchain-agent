"""config 模块单元测试：环境变量加载、校验与不可变性。"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from xiaomi_ocr_agent.config import ConfigError, load_config


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    server = tmp_path / "skills" / "xiaomi-ocr" / "mcp_ocr_server.py"
    server.parent.mkdir(parents=True)
    server.write_text("# stub MCP server\n")
    return tmp_path


def make_env(fake_repo: Path, **overrides: str) -> dict[str, str]:
    env = {
        "OPENAI_BASE_URL": "https://api.example.com/v1",
        "OPENAI_API_KEY": "sk-test",
        "OPENAI_MODEL": "test-model",
        "XIAOMI_OCR_REPO": str(fake_repo),
    }
    env.update(overrides)
    return env


def test_missing_brain_variable_raises_clear_error(fake_repo: Path) -> None:
    env = make_env(fake_repo)
    del env["OPENAI_API_KEY"]
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        load_config(env)


def test_config_is_frozen(fake_repo: Path) -> None:
    cfg = load_config(make_env(fake_repo))
    with pytest.raises(FrozenInstanceError):
        cfg.brain.model = "other"  # type: ignore[misc]


def test_ocr_backend_defaults_target_local_ollama(fake_repo: Path) -> None:
    cfg = load_config(make_env(fake_repo))
    assert cfg.ocr.local_url == "http://127.0.0.1:11434/v1"
    assert cfg.ocr.model == "longwayxu/xiaomi-ocr-0:bf16"
    assert cfg.ocr.mcp_python == fake_repo / ".venv" / "bin" / "python"
    assert cfg.ocr.max_tokens is None


def test_ocr_backend_overrides_are_respected(fake_repo: Path) -> None:
    env = make_env(
        fake_repo,
        XIAOMI_OCR_MODEL="custom-model:tag",
        XIAOMI_OCR_LOCAL_URL="http://localhost:8000/v1",
        XIAOMI_OCR_MAX_TOKENS="2048",
    )
    cfg = load_config(env)
    assert cfg.ocr.model == "custom-model:tag"
    assert cfg.ocr.local_url == "http://localhost:8000/v1"
    assert cfg.ocr.max_tokens == 2048


def test_non_loopback_ocr_url_is_rejected(fake_repo: Path) -> None:
    env = make_env(fake_repo, XIAOMI_OCR_LOCAL_URL="http://192.168.1.5:8000/v1")
    with pytest.raises(ConfigError, match="127.0.0.1"):
        load_config(env)


def test_repo_without_mcp_server_is_rejected(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    with pytest.raises(ConfigError, match="mcp_ocr_server.py"):
        load_config(env)


def test_invalid_max_tokens_is_rejected(fake_repo: Path) -> None:
    env = make_env(fake_repo, XIAOMI_OCR_MAX_TOKENS="not-a-number")
    with pytest.raises(ConfigError, match="XIAOMI_OCR_MAX_TOKENS"):
        load_config(env)
