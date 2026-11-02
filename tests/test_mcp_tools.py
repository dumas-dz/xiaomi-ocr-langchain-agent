"""mcp_tools 模块单元测试：MCP stdio 服务端配置构造（不启动子进程）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from xiaomi_ocr_agent.config import load_config
from xiaomi_ocr_agent.mcp_tools import OCR_TOOL_NAMES, build_mcp_server_config


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


def test_server_config_points_at_mcp_server_script(fake_repo: Path) -> None:
    cfg = load_config(make_env(fake_repo))
    server = build_mcp_server_config(cfg.ocr, base_env={})
    assert server["transport"] == "stdio"
    assert server["command"] == str(fake_repo / ".venv" / "bin" / "python")
    assert server["args"] == [str(fake_repo / "skills" / "xiaomi-ocr" / "mcp_ocr_server.py")]


def test_server_env_carries_xiaomi_ocr_variables(fake_repo: Path) -> None:
    env = make_env(fake_repo, XIAOMI_OCR_MODEL="custom:tag")
    cfg = load_config(env)
    server = build_mcp_server_config(cfg.ocr, base_env={})
    assert server["env"]["XIAOMI_OCR_LOCAL_URL"] == "http://127.0.0.1:11434/v1"
    assert server["env"]["XIAOMI_OCR_MODEL"] == "custom:tag"
    assert server["env"]["XIAOMI_OCR_REPO"] == str(fake_repo)


def test_server_env_inherits_only_allowlisted_variables(fake_repo: Path) -> None:
    cfg = load_config(make_env(fake_repo))
    base_env = {"PATH": "/usr/bin", "OPENAI_API_KEY": "sk-leak", "HOME": "/home/u"}
    server = build_mcp_server_config(cfg.ocr, base_env=base_env)
    assert server["env"]["PATH"] == "/usr/bin"
    assert server["env"]["HOME"] == "/home/u"
    assert "OPENAI_API_KEY" not in server["env"]


def test_max_tokens_is_forwarded_when_set(fake_repo: Path) -> None:
    env = make_env(fake_repo, XIAOMI_OCR_MAX_TOKENS="2048")
    cfg = load_config(env)
    server = build_mcp_server_config(cfg.ocr, base_env={})
    assert server["env"]["XIAOMI_OCR_MAX_TOKENS"] == "2048"


def test_tool_names_match_official_mcp_surface() -> None:
    assert OCR_TOOL_NAMES == ("ocr_image", "ocr_pdf", "kie_image", "vqa_image")
