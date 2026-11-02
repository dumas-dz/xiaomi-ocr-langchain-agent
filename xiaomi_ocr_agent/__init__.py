"""LangChain Agent demo：通过 MCP 调用本地 Xiaomi-OCR-0。"""
from __future__ import annotations

from xiaomi_ocr_agent.agent import build_agent, build_brain
from xiaomi_ocr_agent.config import AppConfig, BrainConfig, ConfigError, OcrConfig, load_config
from xiaomi_ocr_agent.mcp_tools import OCR_TOOL_NAMES, load_ocr_tools

__all__ = [
    "AppConfig",
    "BrainConfig",
    "ConfigError",
    "OCR_TOOL_NAMES",
    "OcrConfig",
    "build_agent",
    "build_brain",
    "load_config",
    "load_ocr_tools",
]
