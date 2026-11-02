"""配置加载：从环境变量构建不可变的 AppConfig。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
REQUIRED_ENV_KEYS = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL", "XIAOMI_OCR_REPO")
DEFAULT_OCR_LOCAL_URL = "http://127.0.0.1:11434/v1"
DEFAULT_OCR_MODEL = "longwayxu/xiaomi-ocr-0:bf16"
MCP_SERVER_RELATIVE_PATH = Path("skills/xiaomi-ocr/mcp_ocr_server.py")


class ConfigError(RuntimeError):
    """环境变量缺失或非法；消息中包含出问题的变量名。"""


@dataclass(frozen=True)
class BrainConfig:
    """Agent 大脑（任意 OpenAI 兼容 API）。"""

    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class OcrConfig:
    """本地 Xiaomi-OCR-0 推理后端与 MCP 服务端参数。"""

    local_url: str
    model: str
    repo_root: Path
    mcp_python: Path
    max_tokens: int | None


@dataclass(frozen=True)
class AppConfig:
    brain: BrainConfig
    ocr: OcrConfig


def load_config(env: Mapping[str, str]) -> AppConfig:
    """校验环境变量并构建 AppConfig；任何缺失/非法项抛 ConfigError。"""
    missing = [key for key in REQUIRED_ENV_KEYS if not env.get(key)]
    if missing:
        raise ConfigError(f"缺少必需的环境变量: {', '.join(missing)}")

    repo_root = Path(env["XIAOMI_OCR_REPO"]).expanduser()
    server_script = repo_root / MCP_SERVER_RELATIVE_PATH
    if not server_script.is_file():
        raise ConfigError(f"XIAOMI_OCR_REPO 下找不到 {MCP_SERVER_RELATIVE_PATH}: {repo_root}")

    local_url = env.get("XIAOMI_OCR_LOCAL_URL", DEFAULT_OCR_LOCAL_URL).strip()
    _validate_loopback(local_url)

    brain = BrainConfig(
        base_url=env["OPENAI_BASE_URL"].strip().rstrip("/"),
        api_key=env["OPENAI_API_KEY"].strip(),
        model=env["OPENAI_MODEL"].strip(),
    )
    ocr = OcrConfig(
        local_url=local_url,
        model=env.get("XIAOMI_OCR_MODEL", DEFAULT_OCR_MODEL).strip(),
        repo_root=repo_root,
        mcp_python=Path(
            env.get("XIAOMI_OCR_MCP_PYTHON", str(repo_root / ".venv" / "bin" / "python"))
        ).expanduser(),
        max_tokens=_parse_max_tokens(env.get("XIAOMI_OCR_MAX_TOKENS")),
    )
    return AppConfig(brain=brain, ocr=ocr)


def _validate_loopback(local_url: str) -> None:
    hostname = urlparse(local_url).hostname
    if hostname not in LOOPBACK_HOSTS:
        raise ConfigError(
            f"XIAOMI_OCR_LOCAL_URL 必须指向本机回环地址（如 http://127.0.0.1:11434/v1），当前: {local_url}"
        )


def _parse_max_tokens(raw: str | None) -> int | None:
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"XIAOMI_OCR_MAX_TOKENS 必须是整数，当前: {raw}") from exc
