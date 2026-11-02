"""MCP 工具加载：以 stdio 子进程启动 xiaomi-ocr MCP 服务并取得 LangChain 工具。"""
from __future__ import annotations

import os
from typing import Any, Mapping

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from xiaomi_ocr_agent.config import MCP_SERVER_RELATIVE_PATH, OcrConfig

OCR_TOOL_NAMES = ("ocr_image", "ocr_pdf", "kie_image", "vqa_image")
INHERITED_ENV_KEYS = frozenset({"PATH", "HOME", "USER", "TMPDIR", "TEMP", "LANG", "LC_ALL"})
XIAOMI_OCR_ENV_PREFIX = "XIAOMI_OCR_"
SERVER_NAME = "xiaomi-ocr"


def build_mcp_server_config(ocr: OcrConfig, base_env: Mapping[str, str]) -> dict[str, Any]:
    """构造 stdio MCP 服务端配置；只透传白名单变量与 XIAOMI_OCR_* 前缀变量。"""
    inherited = {
        key: value
        for key, value in base_env.items()
        if key in INHERITED_ENV_KEYS or key.startswith(XIAOMI_OCR_ENV_PREFIX)
    }
    env: dict[str, str] = {
        **inherited,
        "XIAOMI_OCR_LOCAL_URL": ocr.local_url,
        "XIAOMI_OCR_MODEL": ocr.model,
        "XIAOMI_OCR_REPO": str(ocr.repo_root),
    }
    if ocr.max_tokens is not None:
        env["XIAOMI_OCR_MAX_TOKENS"] = str(ocr.max_tokens)
    return {
        "transport": "stdio",
        "command": str(ocr.mcp_python),
        "args": [str(ocr.repo_root / MCP_SERVER_RELATIVE_PATH)],
        "env": env,
    }


async def load_ocr_tools(ocr: OcrConfig) -> list[BaseTool]:
    """启动 MCP 服务子进程并返回其 4 个 OCR 工具。"""
    server_config = build_mcp_server_config(ocr, base_env=dict(os.environ))
    client = MultiServerMCPClient({SERVER_NAME: server_config})
    tools = await client.get_tools()
    names = sorted(tool.name for tool in tools)
    if names != sorted(OCR_TOOL_NAMES):
        raise RuntimeError(f"MCP 工具集与预期不符: {names} != {sorted(OCR_TOOL_NAMES)}")
    return list(tools)
