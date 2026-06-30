# Aqara API Adapter

本地格式转换代理，将 Anthropic Messages / Responses API 请求转换为 OpenAI Chat Completions 格式，转发到 [Aqara AI Infra](https://ai-infra.aqara.com) 后端。

## 解决的问题

公司 API 网关 (ai-infra.aqara.com) 只支持 OpenAI Chat Completions 格式，但 Claude Code / Claude Desktop / Codex 发出的是 Anthropic Messages 或 Responses API 格式。

## 支持的路由

| 路由 | 格式转换 | 适用工具 |
|------|---------|---------|
| `POST /v1/messages` | Anthropic Messages → Chat Completions | Claude Code / Claude Desktop |
| `POST /v1/responses` | Responses API → Chat Completions | Codex App |
| `POST /v1/chat/completions` | 透传 | Codex CLI / Pi |

## 快速开始

```bash
# 1. 设置 API Key
export AQARA_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# 2. 启动适配器（默认端口 18080）
python3 aqara-api-adapter.py

# 自定义端口和后端
python3 aqara-api-adapter.py --port 18088 --backend https://your-gateway.com/v1
```

## 工具配置

- **Claude Code CLI**: `~/.claude/settings.json` → `env.ANTHROPIC_BASE_URL = "http://127.0.0.1:18080"`
- **Claude Desktop**: 通过 CC Switch 代理 → 适配器 `http://127.0.0.1:18090`
- **Codex CLI**: `~/.codex/config.toml` → `base_url = "http://127.0.0.1:18080"`, `wire_api = "responses"`
- **Pi**: `~/.pi/agent/models.json` → openai-completions provider, baseUrl 指向适配器

## 开机自启 (macOS launchd)

```bash
# 安装
cp launchd/com.aqara.adapter-18090.plist ~/Library/LaunchAgents/
cp launchd/com.aqara.adapter-18080.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.aqara.adapter-18090.plist
launchctl load ~/Library/LaunchAgents/com.aqara.adapter-18080.plist

# 查看状态
launchctl list | grep aqara

# 停止/启动
launchctl unload ~/Library/LaunchAgents/com.aqara.adapter-18090.plist
launchctl load   ~/Library/LaunchAgents/com.aqara.adapter-18090.plist
```

## 模型映射

默认映射（可在 `MODEL_MAP` 字典中修改）：

| Claude 模型名 | 实际模型 |
|--------------|---------|
| `claude-opus-4-7` | `claude-opus-4-6` |
| `claude-opus-4-6` | `claude-opus-4-6` |
| `claude-sonnet-4-6` | `claude-sonnet-4-6` |
| `claude-haiku-4-5` | `deepseek-v4-flash` |

## 依赖

- Python 3.9+
- curl（macOS 自带）
- 零第三方 pip 包

## License

MIT
