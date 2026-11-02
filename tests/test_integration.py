"""集成测试：真实启动 xiaomi-ocr MCP 子进程并调用 Ollama 后端。

运行前置条件：
- Ollama 服务在 http://127.0.0.1:11434/v1 且已拉取 longwayxu/xiaomi-ocr-0:bf16
- 仓库 .venv 可运行 skills/xiaomi-ocr/mcp_ocr_server.py

标记 integration；如只想跑单元测试：pytest -m "not integration"
"""
from __future__ import annotations

from pathlib import Path

import pytest

from xiaomi_ocr_agent.agent import extract_text
from xiaomi_ocr_agent.config import load_config
from xiaomi_ocr_agent.mcp_tools import OCR_TOOL_NAMES, load_ocr_tools

pytestmark = pytest.mark.integration

REPO = Path("/Users/Dumas/WorkSpaces/GitHubProjects/Xiaomi-OCR-0")
EXAMPLE_PNG = REPO / "example_pics" / "receipt.jpg"


def real_env(**overrides: str) -> dict[str, str]:
    env = {
        "OPENAI_BASE_URL": "https://api.example.com/v1",
        "OPENAI_API_KEY": "sk-test",
        "OPENAI_MODEL": "test-model",
        "XIAOMI_OCR_REPO": str(REPO),
    }
    env.update(overrides)
    return env


async def test_load_ocr_tools_exposes_official_surface() -> None:
    cfg = load_config(real_env())
    tools = await load_ocr_tools(cfg.ocr)
    assert sorted(t.name for t in tools) == sorted(OCR_TOOL_NAMES)


async def test_ocr_image_parses_real_receipt() -> None:
    cfg = load_config(real_env())
    tools = {t.name: t for t in await load_ocr_tools(cfg.ocr)}
    raw = await tools["ocr_image"].ainvoke({"image_path": str(EXAMPLE_PNG)})
    result = extract_text(raw)
    assert isinstance(result, str)
    assert len(result) > 200
    assert "receipt" in result.lower() or "发票" in result or "合计" in result
