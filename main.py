"""命令行入口：用自然语言任务驱动 Agent 调用小米 OCR。"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from xiaomi_ocr_agent.agent import build_agent, build_user_message, extract_text
from xiaomi_ocr_agent.config import AppConfig, ConfigError, load_config
from xiaomi_ocr_agent.mcp_tools import load_ocr_tools

logger = logging.getLogger("xiaomi_ocr_agent")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangChain Agent → xiaomi-ocr MCP → 本地 Xiaomi-OCR-0（Ollama）"
    )
    parser.add_argument("task", help="自然语言任务，例如：把这张图片解析成 Markdown")
    parser.add_argument("--image", type=Path, default=None, help="本地图片或 PDF 路径")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="环境变量文件")
    return parser.parse_args(argv)


async def run(cfg: AppConfig, task: str, image: Path | None) -> str:
    """加载 MCP 工具、组装 Agent、执行任务并返回最终回答。"""
    tools = await load_ocr_tools(cfg.ocr)
    agent = build_agent(cfg.brain, tools)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": build_user_message(task, image)}]}
    )
    messages = result["messages"]
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            logger.info("→ 调用工具 %s(%s)", call["name"], call["args"])
    return extract_text(messages[-1].content)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    load_dotenv(args.env_file)

    image: Path | None = None
    if args.image is not None:
        image = args.image.expanduser()
        if not image.is_file():
            logger.error("图片/PDF 不存在: %s", image)
            return 2

    try:
        cfg = load_config(os.environ)
        answer = asyncio.run(run(cfg, args.task, image))
    except ConfigError as exc:
        logger.error("配置错误: %s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 — CLI 顶层兜底，显式记录并返回非零
        logger.error("执行失败: %s", exc)
        return 1

    logger.info(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
