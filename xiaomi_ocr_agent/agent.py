"""Agent 组装：OpenAI 兼容大脑 + MCP OCR 工具。"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from xiaomi_ocr_agent.config import BrainConfig

SYSTEM_PROMPT = (
    "你是 Xiaomi-OCR-0 文档解析助手，通过 MCP 工具调用本地模型完成任务。\n"
    "可用工具与选择规则：\n"
    "- 文档解析/OCR：ocr_image。印刷规整的文档用 mode='region'（自动版面检测、分区并发识别）；"
    "手写笔记、场景文字、书法、古籍或版式不规则的材料用 mode='page'，避免区域切分破坏上下文。\n"
    "- PDF 解析：ocr_pdf，模式选择规则同上。\n"
    "- 关键信息抽取（KIE）：kie_image，已知字段名时通过 fields 参数传入。\n"
    "- 文档图像问答（VQA）：vqa_image。\n"
    "工具返回的 Markdown/JSON 即最终结果：原样呈现给用户，不要改写、补全或臆造识别内容。"
)


def build_brain(brain: BrainConfig) -> ChatOpenAI:
    """构造 OpenAI 兼容对话模型（temperature=0，保证可复现）。"""
    return ChatOpenAI(
        model=brain.model,
        base_url=brain.base_url,
        api_key=brain.api_key,
        temperature=0,
    )


def build_agent(brain: BrainConfig, tools: Sequence[Any]) -> CompiledStateGraph:
    """把大脑与工具装配成一个工具调用 Agent。"""
    return create_agent(build_brain(brain), tools, system_prompt=SYSTEM_PROMPT)


def build_user_message(task: str, image: Path | None) -> str:
    """拼装用户消息；附带本地文件路径，供大脑填入工具的 image_path/pdf_path。"""
    if image is None:
        return task
    return f"{task}\n\n本地文件路径（传给工具的 image_path/pdf_path 参数）: {image.expanduser().resolve()}"


def extract_text(content: object) -> str:
    """把消息内容规整为纯文本；兼容 str 与 MCP 内容块列表。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block.get("text", "") if isinstance(block, dict) else str(block) for block in content]
        return "".join(parts)
    return str(content)
