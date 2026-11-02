"""agent 模块单元测试：大脑构造、系统提示词与工具装配（不发真实网络请求）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from xiaomi_ocr_agent.agent import SYSTEM_PROMPT, build_agent, build_brain, extract_text
from xiaomi_ocr_agent.config import load_config

pytestmark = pytest.mark.unit


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


def test_build_brain_uses_openai_compatible_endpoint(fake_repo: Path) -> None:
    cfg = load_config(make_env(fake_repo))
    brain = build_brain(cfg.brain)
    assert brain.model_name == "test-model"
    assert str(brain.openai_api_base).rstrip("/").endswith("api.example.com/v1")


def test_build_agent_registers_all_ocr_tools(fake_repo: Path) -> None:
    cfg = load_config(make_env(fake_repo))

    def ocr_image(image_path: str) -> str:
        """Parse one image to Markdown."""

    def ocr_pdf(pdf_path: str) -> str:
        """Parse one PDF to Markdown."""

    def kie_image(image_path: str, fields: list[str] | None = None) -> str:
        """Extract key information as JSON."""

    def vqa_image(image_path: str, question: str) -> str:
        """Answer a question about a document image."""

    agent = build_agent(cfg.brain, [ocr_image, ocr_pdf, kie_image, vqa_image])
    assert agent is not None


def test_system_prompt_guides_task_and_mode_choice() -> None:
    assert "ocr_image" in SYSTEM_PROMPT
    assert "kie_image" in SYSTEM_PROMPT
    assert "vqa_image" in SYSTEM_PROMPT
    assert "region" in SYSTEM_PROMPT
    assert "page" in SYSTEM_PROMPT


def test_extract_text_handles_string_content() -> None:
    assert extract_text("hello") == "hello"


def test_extract_text_joins_mcp_content_blocks() -> None:
    blocks = [{"type": "text", "text": "第一行"}, {"type": "text", "text": "第二行"}]
    assert extract_text(blocks) == "第一行第二行"


def test_extract_text_falls_back_to_str() -> None:
    assert extract_text(42) == "42"
