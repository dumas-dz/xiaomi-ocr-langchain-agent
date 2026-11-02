# Xiaomi-OCR-0 × LangChain Agent Demo

用 LangChain 搭一个简单 Agent，通过 **MCP** 调用本地 **Xiaomi-OCR-0**（Ollama 部署）完成文档解析、KIE 和 VQA。

```text
自然语言任务
   │
   ▼
LangChain Agent（大脑 = 任意 OpenAI 兼容 API，.env 配置）
   │  工具调用决策
   ▼
xiaomi-ocr MCP 服务（stdio 子进程，官方 skills/xiaomi-ocr/mcp_ocr_server.py）
   │  提示词 / 预处理 / 可选区域检测 / 后处理
   ▼
Ollama http://127.0.0.1:11434/v1  ←  longwayxu/xiaomi-ocr-0:bf16
```

Agent 可用的 4 个 MCP 工具：`ocr_image`（整页/区域文档解析）、`ocr_pdf`、`kie_image`（关键信息抽取）、`vqa_image`（文档问答）。

## 前置条件

1. Ollama 服务运行中，且已拉取模型：
   ```bash
   ollama pull longwayxu/xiaomi-ocr-0:bf16
   ```
2. 本机克隆了 [Xiaomi-OCR-0 仓库](https://github.com/SeerRay-Lab/Xiaomi-OCR-0)，其 `.venv` 能运行 `skills/xiaomi-ocr/mcp_ocr_server.py`（依赖 `Pillow/pypdfium2/numpy/tqdm/wordfreq`）。

## 安装与配置

```bash
cd xiaomi-ocr-langchain-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # 填入你的 OpenAI 兼容 API 的 BASE_URL / KEY / MODEL
```

`.env` 中 `XIAOMI_OCR_*` 默认指向本机 Ollama 与仓库路径，如路径不同请相应修改。MCP 子进程只允许回环地址（`127.0.0.1`），图片不会外发。

## 运行

```bash
# 文档解析（Agent 自动选择 ocr_image 与 page/region 模式）
python main.py "把这张小票解析成 Markdown" --image /path/to/receipt.jpg

# 关键信息抽取（Agent 自动选择 kie_image）
python main.py "提取这张发票的金额、日期和商家" --image /path/to/invoice.jpg

# 文档问答（Agent 自动选择 vqa_image）
python main.py "图片里一共消费了多少钱？" --image /path/to/receipt.jpg

# PDF 解析（Agent 自动选择 ocr_pdf）
python main.py "解析这份 PDF 的内容" --image /path/to/doc.pdf
```

运行日志会显示 Agent 的工具调用轨迹（`→ 调用工具 ...`），最后输出识别结果。

## 测试

```bash
pytest -m "not integration"   # 单元测试（不联网、不起子进程）
pytest                        # 全部，含集成测试（真实 MCP + Ollama）
```

集成测试会真实启动 MCP 子进程并对 `example_pics/receipt.jpg` 做一次 OCR，需要 Ollama 在线。

## 设计要点

- **配置**（`xiaomi_ocr_agent/config.py`）：冻结 dataclass + 启动即校验，`XIAOMI_OCR_LOCAL_URL` 强制回环地址。
- **工具加载**（`xiaomi_ocr_agent/mcp_tools.py`）：`langchain-mcp-adapters` 的 `MultiServerMCPClient` 以 stdio 启动官方 MCP 服务；子进程环境只透传白名单变量与 `XIAOMI_OCR_*`，API Key 不会泄漏给 MCP 进程。
- **Agent**（`xiaomi_ocr_agent/agent.py`）：`langchain.agents.create_agent` 装配工具调用 Agent，system prompt 内置官方任务/模式选择规则（印刷文档 region、手写/古籍 page）。
- 图像预处理、提示词、OTSL→Markdown 表格转换等全部留在官方 MCP 服务内部，Agent 只做任务路由，不重复实现识别逻辑。
