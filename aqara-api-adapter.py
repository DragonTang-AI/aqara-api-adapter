#!/usr/bin/env python3
"""
============================================================================
 Aqara AI API 适配器 — Anthropic Messages / Responses API → OpenAI Chat Completions
============================================================================
  解决的问题：公司 API 网关 (ai-infra.aqara.com) 只支持 OpenAI Chat Completions，
  但 Claude Code / Codex 发出的是 Anthropic Messages 或 Responses API 格式。

  这个适配器在本地启动一个 HTTP 服务，自动做格式转换 + 中继。

  支持 3 条路由：
    POST /v1/messages          → Anthropic Messages → Chat Completions  (Claude Code)
    POST /v1/responses         → Responses API    → Chat Completions  (Codex App)
    POST /v1/chat/completions  → 透传                               (Codex CLI, Pi 等)

  用法:
    # 1. 设置 API Key
    export AQARA_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"

    # 2. 启动适配器（默认端口 18080）
    python3 aqara-api-adapter.py

    # 自定义端口和后端
    python3 aqara-api-adapter.py --port 18088 --backend https://your-gateway.com/v1

    # 查看帮助
    python3 aqara-api-adapter.py --help

  工具配置:
    Claude Code CLI  → settings.json env.ANTHROPIC_BASE_URL = "http://127.0.0.1:18080"
    Codex CLI        → config.toml base_url = "http://127.0.0.1:18080", wire_api = "responses"
    Pi               → models.json 配 openai-completions provider，baseUrl 指向适配器
    CC Switch        → 填 http://127.0.0.1:18080 为后端地址

  依赖: Python 3.9+, curl（macOS 自带）, 无第三方 pip 包
============================================================================
"""

import argparse
import json
import http.server
import socketserver
import subprocess
import sys
import os
import threading
from datetime import datetime


# ── 默认配置 ──────────────────────────────────────────────────────
DEFAULT_PORT = 18080
DEFAULT_BACKEND = "https://ai-infra.aqara.com/v1"

# 模型映射：Claude 模型名 → 公司实际模型名
# 可以根据需要修改
MODEL_MAP = {
    "claude-sonnet-4-6":        "claude-sonnet-4-6",
    "claude-opus-4-7":          "claude-opus-4-6",
    "claude-opus-4-6":          "claude-opus-4-6",
    "claude-haiku-4-5":         "deepseek-v4-flash",
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-6",
    "claude-3-5-haiku-20241022":  "deepseek-v4-flash",
}

# 所有可用模型（供 /v1/models 返回）
AVAILABLE_MODELS = [
    "claude-opus-4-6", "claude-sonnet-4-6", "deepseek-v4-flash",
    "deepseek-v4-pro", "gemini-3-flash-preview", "gemini-3.1-pro-preview",
    "gemini-3.5-flash", "glm-5.1", "gpt-4.1-nano", "gpt-4o-mini",
    "gpt-5.1-codex-max", "gpt-5.1-codex-mini", "gpt-5.2", "gpt-5.2-codex",
    "gpt-5.3-codex", "gpt-5.4", "gpt-5.4-mini", "gpt-5.5", "kimi-k2.6",
    "qwen3.6-plus",
]


# ── 日志 ──────────────────────────────────────────────────────────

def _make_logger(log_file):
    """创建一个简单的日志函数"""
    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        if log_file:
            try:
                with open(log_file, "a") as f:
                    f.write(line + "\n")
            except Exception:
                pass
    return log

log = print  # 占位，main 里替换


# ── 后端调用 ──────────────────────────────────────────────────────

def curl_call(body_json, api_key, backend_url, stream=False, timeout=300):
    """同步 curl 调用后端，返回 (returncode, stdout, stderr)"""
    cmd = [
        "curl", "-s", "-k", "--noproxy", "*", "--show-error",
        "--max-time", str(timeout),
        "-X", "POST",
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Content-Type: application/json",
        "-d", body_json,
    ]
    cmd += ["-N", "--no-buffer"]
    cmd.append(f"{backend_url}/chat/completions")

    p = subprocess.run(cmd, capture_output=True, timeout=timeout + 10)
    return p.returncode, p.stdout, p.stderr


def curl_stream(body_json, api_key, backend_url, timeout=300):
    """流式 curl 调用，逐行 yield SSE data line"""
    cmd = [
        "curl", "-s", "-k", "--noproxy", "*", "--show-error",
        "--max-time", str(timeout),
        "-N", "--no-buffer",
        "-X", "POST",
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Content-Type: application/json",
        "-d", body_json,
        f"{backend_url}/chat/completions"
    ]
    log(f"  curl-stream: {len(body_json)} bytes, timeout={timeout}s")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    leftover = b""
    while True:
        chunk = p.stdout.read(4096)
        if not chunk:
            break
        data = leftover + chunk
        lines = data.split(b"\n")
        leftover = lines.pop()
        for line in lines:
            line = line.strip()
            if line.startswith(b"data: "):
                yield line[6:]
            elif line in (b"data:", b"data: "):
                pass

    if leftover.strip().startswith(b"data: "):
        yield leftover.strip()[6:]

    p.wait()
    if p.returncode != 0:
        err = p.stderr.read().decode(errors="replace")
        log(f"  curl error (exit={p.returncode}): {err[:200]}")


# ── 格式转换：Anthropic → OpenAI ──────────────────────────────────

def convert_anthropic_to_openai(anthropic_body, model_map):
    """Anthropic Messages 请求 → OpenAI Chat Completions 请求"""
    model = anthropic_body.get("model", "deepseek-v4-flash")
    model = model_map.get(model, model)
    log(f"  model: {anthropic_body.get('model','?')} -> {model}")

    max_tokens = anthropic_body.get("max_tokens", 4096)
    stream = anthropic_body.get("stream", False)
    temperature = anthropic_body.get("temperature", 1.0)
    top_p = anthropic_body.get("top_p")
    top_k = anthropic_body.get("top_k")
    stop_sequences = anthropic_body.get("stop_sequences")
    system = anthropic_body.get("system", "")
    messages_in = anthropic_body.get("messages", [])

    openai_messages = []
    if system:
        if isinstance(system, str):
            openai_messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            txt = "".join(
                s.get("text", "")
                for s in system
                if isinstance(s, dict) and s.get("type") == "text"
            )
            if txt:
                openai_messages.append({"role": "system", "content": txt})

    for msg in messages_in:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "image":
                        source = block.get("source", {})
                        mt = source.get("media_type", "image/png")
                        b64 = source.get("data", "")
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mt};base64,{b64}"}
                        })
            content = "\n".join(parts) if all(isinstance(p, str) for p in parts) else parts
        else:
            content = str(content)
        openai_messages.append({"role": role, "content": content})

    # Anthropic tools → OpenAI tools
    anthropic_tools = anthropic_body.get("tools", [])
    openai_tools = []
    for tool in anthropic_tools:
        if isinstance(tool, dict):
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema",
                                          {"type": "object", "properties": {}}),
                }
            })

    tool_choice = anthropic_body.get("tool_choice")

    body = {
        "model": model,
        "messages": openai_messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": temperature,
    }
    if openai_tools:
        body["tools"] = openai_tools
        if tool_choice:
            if isinstance(tool_choice, dict):
                tc_type = tool_choice.get("type", "")
                if tc_type == "any":
                    body["tool_choice"] = "required"
                elif tc_type == "auto":
                    body["tool_choice"] = "auto"
                elif tc_type == "tool":
                    body["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_choice.get("name", "")}
                    }
            elif tool_choice == "any":
                body["tool_choice"] = "required"

    if top_p is not None:
        body["top_p"] = top_p
    if stop_sequences:
        body["stop"] = stop_sequences

    # top_k 不是 OpenAI 标准参数，放到 extra_body
    if top_k is not None:
        body["extra_body"] = {"top_k": top_k}

    return body


# ── 格式转换：OpenAI → Anthropic ──────────────────────────────────

def convert_openai_to_anthropic(openai_data):
    """OpenAI Chat Completions 响应 → Anthropic Messages 响应"""
    msg_id = openai_data.get("id", "")
    model = openai_data.get("model", "")
    if not openai_data.get("choices"):
        return {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": ""}], "model": model,
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    choice = openai_data["choices"][0]
    msg = choice.get("message", {})
    text = msg.get("content", "") or ""
    finish = choice.get("finish_reason", "stop")
    stop_reason = {
        "stop": "end_turn", "length": "max_tokens",
        "tool_calls": "tool_use",
    }.get(finish, "end_turn")
    usage = openai_data.get("usage", {})

    content = []
    if text:
        content.append({"type": "text", "text": text})

    tool_calls = msg.get("tool_calls", [])
    for tc in tool_calls:
        func = tc.get("function", {})
        content.append({
            "type": "tool_use",
            "id": tc.get("id", f"call_{tc.get('index', 0)}"),
            "name": func.get("name", ""),
            "input": json.loads(func.get("arguments", "{}")),
        })

    return {
        "id": msg_id, "type": "message", "role": "assistant",
        "content": content,
        "model": model, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ── 格式转换：Responses → Chat ────────────────────────────────────

def convert_responses_to_chat(responses_body):
    """Responses API 请求 → Chat Completions 请求"""
    model = responses_body.get("model", "deepseek-v4-flash")
    messages = []

    instructions = responses_body.get("instructions", "")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    inp = responses_body.get("input", "")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            role = item.get("role", "user")
            content_parts = item.get("content", "")
            if isinstance(content_parts, list):
                texts = []
                for c in content_parts:
                    if isinstance(c, dict):
                        t = c.get("type", "")
                        if t in ("input_text", "output_text", "text"):
                            texts.append(c.get("text", ""))
                        elif t == "input_image":
                            texts.append(f"[image: {c.get('image_url', '')}]")
                content = "\n".join(texts)
            elif isinstance(content_parts, str):
                content = content_parts
            else:
                content = str(content_parts)
            messages.append({"role": role, "content": content})

    # Responses API tools → OpenAI tools
    tools = responses_body.get("tools", [])
    openai_tools = []
    for tool in tools:
        if isinstance(tool, dict):
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                }
            })

    max_tokens_val = (
        responses_body.get("max_output_tokens")
        or responses_body.get("max_tokens")
        or 4096
    )
    max_tokens_val = max(max_tokens_val, 4096)

    chat_body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens_val,
        "stream": responses_body.get("stream", False),
        "temperature": responses_body.get("temperature", 1.0),
    }
    if openai_tools:
        chat_body["tools"] = openai_tools

    top_p = responses_body.get("top_p")
    if top_p is not None:
        chat_body["top_p"] = top_p

    stop = responses_body.get("stop")
    if stop:
        chat_body["stop"] = stop

    return chat_body


# ── 格式转换：Chat → Responses ────────────────────────────────────

def convert_chat_to_responses(chat_data, model):
    """Chat Completions 响应 → Responses API 响应（非流式）"""
    msg_id = "resp_" + model.replace(".", "_")
    choice = chat_data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    text = msg.get("content", "")
    usage = chat_data.get("usage", {})

    return {
        "id": msg_id,
        "object": "response",
        "model": model,
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }],
        "usage": {
            "input_tokens": usage.get("prompt_tokens", max(1, len(text) // 2)),
            "output_tokens": usage.get("completion_tokens", max(1, len(text) // 2)),
            "total_tokens": usage.get("total_tokens", max(1, len(text))),
        },
    }


# ── HTTP 服务器 ────────────────────────────────────────────────────

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """多线程 HTTP 服务器，支持并发请求"""
    daemon_threads = True
    allow_reuse_address = True


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # 这些类属性由 main 在创建前设置
    api_key = ""
    backend_url = ""
    model_map = {}
    available_models = []
    log = None

    def log_message(self, fmt, *args):
        pass  # 用自定义日志替代默认日志

    def _log_req(self):
        ct = self.headers.get("Content-Type", "?")
        tn = threading.current_thread().name
        log(f"<-- {self.command} {self.path} [{ct}] thread={tn}")

    def _send_json(self, code, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._log_req()
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    @staticmethod
    def _extract_path(raw_path):
        """从 self.path 提取纯路径。curl 有时发完整 URL"""
        from urllib.parse import urlparse
        parsed = urlparse(raw_path)
        path = parsed.path.rstrip("/") or "/"
        return path

    def do_GET(self):
        self._log_req()
        path = self._extract_path(self.path)

        if path in ("", "/"):
            self._send_json(200, {"status": "ok", "service": "aqara-api-adapter"})
        elif path == "/health":
            self._send_json(200, {"status": "ok", "version": "v2.0-shared"})
        elif path in ("/v1/models", "/v1/models/"):
            models = [
                {"id": m, "type": "model", "display_name": m,
                 "created_at": "2024-01-01T00:00:00Z"}
                for m in self.available_models
            ]
            self._send_json(200, {
                "data": models, "has_more": False,
                "first_id": self.available_models[0],
                "last_id": self.available_models[-1],
            })
        else:
            log(f"  [404 GET] {path}")
            self._send_json(404, {"error": f"Unknown path: {path}"})

    def do_POST(self):
        self._log_req()
        path = self._extract_path(self.path)
        cl = int(self.headers.get("Content-Length", 0))

        if path == "/v1/messages":
            self._handle_messages(cl)
        elif path == "/v1/messages/count_tokens":
            self._handle_count_tokens(cl)
        elif path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_chat_completions(cl)
        elif path in ("/v1/responses", "/responses"):
            self._handle_responses(cl)
        else:
            log(f"  [404 POST] {path}")
            self._send_json(404, {"error": f"Unknown path: {path}"})

    # ── 路由处理器 ────────────────────────────────────────────────

    def _handle_messages(self, cl):
        body = json.loads(self.rfile.read(cl))
        stream = body.get("stream", False)
        openai_body = convert_anthropic_to_openai(body, self.model_map)
        body_json = json.dumps(openai_body)
        log(
            f"  >> [messages] model={openai_body['model']} stream={stream} "
            f"msgs={len(openai_body['messages'])} "
            f"tools={len(body.get('tools',[]))}"
        )
        try:
            if stream:
                self._send_anthropic_stream(body_json, openai_body["model"])
            else:
                self._send_anthropic_sync(body_json, openai_body["model"])
        except Exception as e:
            log(f"  !! [messages] ERROR: {e}")
            self._send_json(500, {
                "type": "error",
                "error": {"type": "server_error", "message": str(e)},
            })

    def _handle_chat_completions(self, cl):
        raw_body = self.rfile.read(cl)
        body = json.loads(raw_body)
        model = body.get("model", "?")
        stream = body.get("stream", False)
        log(f"  >> [proxy] model={model} stream={stream}")
        try:
            if stream:
                self._proxy_stream(raw_body, model)
            else:
                code, out, err = curl_call(
                    raw_body.decode(), self.api_key, self.backend_url, stream=False
                )
                if code != 0:
                    raise Exception(f"curl exit={code}")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                log(f"  << [proxy] OK {len(out)} bytes")
        except Exception as e:
            log(f"  !! [proxy] ERROR: {e}")
            self._send_json(500, {"error": str(e)})

    def _handle_responses(self, cl):
        raw_body = self.rfile.read(cl)
        body = json.loads(raw_body)
        model = body.get("model", "deepseek-v4-flash")
        stream = body.get("stream", False)
        log(
            f"  >> [responses] model={model} stream={stream} "
            f"tools={len(body.get('tools',[]))}"
        )
        chat_body = convert_responses_to_chat(body)
        body_json = json.dumps(chat_body)
        log(
            f"  >> [responses→chat] model={chat_body['model']} "
            f"msgs={len(chat_body['messages'])}"
        )
        try:
            if stream:
                self._send_responses_stream(body_json, model)
            else:
                self._send_responses_sync(body_json, model)
        except Exception as e:
            log(f"  !! [responses] ERROR: {e}")
            self._send_json(500, {"error": str(e)})

    def _handle_count_tokens(self, cl):
        body = json.loads(self.rfile.read(cl))
        chars = 0
        for msg in body.get("messages", []):
            c = msg.get("content", "")
            if isinstance(c, str):
                chars += len(c)
            elif isinstance(c, list):
                chars += sum(
                    len(b.get("text", ""))
                    for b in c
                    if isinstance(b, dict) and b.get("type") == "text"
                )
        est = max(1, chars // 4)
        log(f"  count_tokens: ~{est}")
        self._send_json(200, {"input_tokens": est})

    # ── Anthropic Messages 响应 ────────────────────────────────────

    def _send_anthropic_sync(self, body_json, model):
        code, out, err = curl_call(body_json, self.api_key, self.backend_url, stream=False)
        if code != 0:
            raise Exception(f"curl exit={code}: {err.decode(errors='replace')[:300]}")
        data = json.loads(out)
        result = convert_openai_to_anthropic(data)
        content = result.get("content", [])
        tc = sum(1 for b in content if b.get("type") == "tool_use")
        log(
            f"  << OK {sum(len(b.get('text','')) for b in content if b.get('type')=='text')} "
            f"chars + {tc} tool_uses"
        )
        self._send_json(200, result)

    def _anthro_sse(self, event_type, data):
        """发送单个 Anthropic SSE 事件"""
        self.wfile.write(
            f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()
        )

    def _send_anthropic_stream(self, body_json, model):
        """Anthropic Messages 流式响应 — 完整 content_block 状态机"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        # message_start + ping
        self._anthro_sse("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_001", "type": "message", "role": "assistant",
                "content": [], "model": model, "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })
        self.wfile.write(b"event: ping\ndata: {}\n\n")
        self.wfile.flush()

        # 状态机
        block_idx = 0
        block_open = False
        block_type = None       # "text" | "tool_use"
        open_tool_idx = None    # 当前打开的 OpenAI tool_call index
        tool_calls_acc = {}     # oai_idx → {id, name, partial_json, block_index}
        total_text = 0
        chunk_count = 0
        skip_count = 0
        final_finish_reason = "end_turn"

        def close_block():
            nonlocal block_open, block_type, open_tool_idx, block_idx
            if block_open and block_type is not None:
                self._anthro_sse("content_block_stop", {
                    "type": "content_block_stop", "index": block_idx,
                })
                self.wfile.flush()
                block_open = False
                block_type = None
                open_tool_idx = None
                block_idx += 1

        try:
            for line_data in curl_stream(body_json, self.api_key, self.backend_url, timeout=600):
                chunk_count += 1
                if line_data == b"[DONE]":
                    continue
                try:
                    cd = json.loads(line_data)
                    choices = cd.get("choices", [])
                    if not choices:
                        skip_count += 1
                        continue
                    delta = choices[0].get("delta", {})
                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason:
                        final_finish_reason = finish_reason

                    # tool_calls delta
                    tc_list = delta.get("tool_calls", [])
                    if tc_list:
                        for tc in tc_list:
                            oai_idx = tc.get("index", 0)
                            func = tc.get("function", {})

                            if oai_idx not in tool_calls_acc:
                                close_block()
                                call_id = tc.get("id", f"call_{oai_idx}")
                                call_name = func.get("name", "")
                                tool_calls_acc[oai_idx] = {
                                    "id": call_id, "name": call_name,
                                    "partial_json": "", "block_index": block_idx,
                                }
                                open_tool_idx = oai_idx
                                block_open = True
                                block_type = "tool_use"
                                self._anthro_sse("content_block_start", {
                                    "type": "content_block_start",
                                    "index": block_idx,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": call_id, "name": call_name, "input": {},
                                    },
                                })
                                self.wfile.flush()
                            elif open_tool_idx != oai_idx:
                                close_block()
                                tc_acc = tool_calls_acc[oai_idx]
                                open_tool_idx = oai_idx
                                block_open = True
                                block_type = "tool_use"
                                block_idx = tc_acc["block_index"]
                                self._anthro_sse("content_block_start", {
                                    "type": "content_block_start",
                                    "index": block_idx,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tc_acc["id"],
                                        "name": tc_acc["name"],
                                        "input": {},
                                    },
                                })
                                self.wfile.flush()

                            args = func.get("arguments", "")
                            if args:
                                tool_calls_acc[oai_idx]["partial_json"] += args
                                self._anthro_sse("content_block_delta", {
                                    "type": "content_block_delta",
                                    "index": tool_calls_acc[oai_idx]["block_index"],
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": args,
                                    },
                                })
                                self.wfile.flush()
                        continue

                    # 文本 delta
                    text = delta.get("content", "")
                    if text:
                        if block_type == "tool_use":
                            close_block()
                        if not block_open:
                            block_open = True
                            block_type = "text"
                            self._anthro_sse("content_block_start", {
                                "type": "content_block_start",
                                "index": block_idx,
                                "content_block": {"type": "text", "text": ""},
                            })
                            self.wfile.flush()
                        self._anthro_sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "text_delta", "text": text},
                        })
                        self.wfile.flush()
                        total_text += len(text)
                    else:
                        skip_count += 1
                except json.JSONDecodeError:
                    skip_count += 1
        except Exception as e:
            log(f"  !! Stream error: {e}")

        close_block()

        # message_delta + message_stop
        stop_map = {
            "stop": "end_turn", "length": "max_tokens",
            "tool_calls": "tool_use", "content_filter": "end_turn",
        }
        stop_reason = stop_map.get(final_finish_reason, "end_turn")
        self._anthro_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": max(1, total_text // 4)},
        })
        self.wfile.write(b'event: message_stop\ndata: {"type": "message_stop"}\n\n')
        self.wfile.flush()

        log(
            f"  << Stream OK text={total_text} chars blocks={block_idx} "
            f"stop={stop_reason} tools={len(tool_calls_acc)}"
        )

    # ── Responses API 响应 ──────────────────────────────────────────

    def _send_responses_sync(self, body_json, model):
        code, out, err = curl_call(body_json, self.api_key, self.backend_url, stream=False)
        if code != 0:
            raise Exception(f"curl exit={code}")
        chat_data = json.loads(out)
        resp_data = convert_chat_to_responses(chat_data, model)
        log("  << [responses] OK")
        self._send_json(200, resp_data)

    def _send_responses_stream(self, body_json, model):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        resp_id = "resp_" + model.replace(".", "_").replace("-", "_")
        msg_item_id = resp_id + "_msg"

        # ① response.created + in_progress
        self.wfile.write(
            f'event: response.created\ndata: {{"type":"response.created",'
            f'"response":{{"id":"{resp_id}","object":"response",'
            f'"model":"{model}","output":[]}}}}\n\n'.encode()
        )
        self.wfile.write(
            f'event: response.in_progress\ndata: {{"type":"response.in_progress",'
            f'"response":{{"id":"{resp_id}","object":"response"}}}}\n\n'.encode()
        )
        self.wfile.flush()

        # ② output_item.added + content_part.added
        item_event = {
            "type": "response.output_item.added", "output_index": 0,
            "item": {
                "id": msg_item_id, "type": "message",
                "role": "assistant", "content": [],
            },
        }
        self.wfile.write(f"data: {json.dumps(item_event)}\n\n".encode())
        part_event = {
            "type": "response.content_part.added",
            "item_id": msg_item_id, "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        }
        self.wfile.write(f"data: {json.dumps(part_event)}\n\n".encode())
        self.wfile.flush()

        # ③ 流式处理
        total_text = ""
        chunk_count = 0
        skip_count = 0
        has_tool_call = False
        tool_calls_acc = {}

        try:
            for line_data in curl_stream(body_json, self.api_key, self.backend_url, timeout=600):
                chunk_count += 1
                if line_data == b"[DONE]":
                    continue
                try:
                    cd = json.loads(line_data)
                    choices = cd.get("choices", [])
                    if not choices:
                        skip_count += 1
                        continue
                    delta = choices[0].get("delta", {})

                    # tool_calls delta
                    tc_list = delta.get("tool_calls", [])
                    if tc_list:
                        has_tool_call = True
                        for tc in tc_list:
                            idx = tc.get("index", 0)
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {
                                    "id": tc.get("id", f"call_{idx}"),
                                    "name": "", "args": "",
                                }
                            func = tc.get("function", {})
                            if func.get("name"):
                                tool_calls_acc[idx]["name"] = func["name"]
                                tool_calls_acc[idx]["id"] = tc.get(
                                    "id", tool_calls_acc[idx]["id"]
                                )
                            args_delta = func.get("arguments", "")
                            if args_delta:
                                tool_calls_acc[idx]["args"] += args_delta
                                fcd = {
                                    "type": "response.function_call_arguments.delta",
                                    "item_id": msg_item_id, "output_index": 0,
                                    "call_id": tool_calls_acc[idx]["id"],
                                    "delta": args_delta,
                                }
                                self.wfile.write(
                                    f"data: {json.dumps(fcd)}\n\n".encode()
                                )
                                self.wfile.flush()
                        continue

                    # 文本 delta
                    text = delta.get("content", "")
                    if text:
                        sse = self._chat_chunk_to_responses_sse(cd, resp_id, msg_item_id)
                        if sse:
                            self.wfile.write(sse.encode())
                            self.wfile.flush()
                            total_text += text
                    else:
                        skip_count += 1
                except json.JSONDecodeError:
                    skip_count += 1
        except Exception as e:
            log(f"  !! [responses stream] error: {e}")

        log(
            f"  << [responses stream] chunks={chunk_count} skipped={skip_count} "
            f"text={len(total_text)} chars tools={has_tool_call}"
        )

        # ④ 收尾事件
        if has_tool_call:
            for idx in sorted(tool_calls_acc.keys()):
                tc = tool_calls_acc[idx]
                fcd_done = {
                    "type": "response.function_call_arguments.done",
                    "item_id": msg_item_id, "output_index": 0,
                    "call_id": tc["id"], "name": tc["name"],
                    "arguments": tc["args"],
                }
                self.wfile.write(f"data: {json.dumps(fcd_done)}\n\n".encode())
            self.wfile.flush()

        # output_text.done
        done_event = {
            "type": "response.output_text.done",
            "item_id": msg_item_id, "output_index": 0, "content_index": 0,
            "text": total_text,
        }
        self.wfile.write(f"data: {json.dumps(done_event)}\n\n".encode())

        # output_item.done
        item_content = [{"type": "output_text", "text": total_text}]
        if has_tool_call:
            for idx in sorted(tool_calls_acc.keys()):
                tc = tool_calls_acc[idx]
                item_content.append({
                    "type": "function_call", "id": tc["id"],
                    "call_id": tc["id"], "name": tc["name"],
                    "arguments": tc["args"],
                })
        item_done = {
            "type": "response.output_item.done", "output_index": 0,
            "item": {
                "id": msg_item_id, "type": "message",
                "role": "assistant", "content": item_content,
            },
        }
        self.wfile.write(f"data: {json.dumps(item_done)}\n\n".encode())

        # response.completed
        output = [{
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": total_text}],
        }]
        if has_tool_call:
            for idx in sorted(tool_calls_acc.keys()):
                tc = tool_calls_acc[idx]
                output.append({
                    "type": "function_call", "id": tc["id"],
                    "call_id": tc["id"], "name": tc["name"],
                    "arguments": tc["args"],
                })
        completion = {
            "type": "response.completed",
            "response": {
                "id": resp_id, "object": "response", "model": model,
                "output": output,
                "usage": {
                    "input_tokens": max(1, len(total_text) // 2),
                    "output_tokens": max(1, len(total_text) // 2),
                    "total_tokens": max(1, len(total_text)),
                },
            },
        }
        self.wfile.write(f"data: {json.dumps(completion)}\n\n".encode())
        self.wfile.flush()

    # ── 透传代理 ────────────────────────────────────────────────────

    def _proxy_stream(self, body_bytes, model):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.wfile.flush()

        body_json = body_bytes.decode()
        total = 0
        try:
            for line_data in curl_stream(body_json, self.api_key, self.backend_url, timeout=600):
                self.wfile.write(b"data: " + line_data + b"\n\n")
                self.wfile.flush()
                total += len(line_data)
        except Exception as e:
            log(f"  !! [proxy stream] error: {e}")
        log(f"  << [proxy stream] OK ~{total} bytes")

    @staticmethod
    def _chat_chunk_to_responses_sse(chunk_data, resp_id, item_id=None):
        if not chunk_data.get("choices"):
            return None
        choice = chunk_data["choices"][0]
        delta = choice.get("delta", {})
        text = delta.get("content", "")
        if not text:
            return None
        event = {
            "type": "response.output_text.delta",
            "item_id": item_id or (resp_id + "_msg"),
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        }
        return f"data: {json.dumps(event)}\n\n"


# ── main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aqara AI API 适配器 — Anthropic/Responses API → OpenAI Chat Completions 格式转换代理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  export AQARA_API_KEY="sk-xxxx"
  python3 aqara-api-adapter.py                           # 默认 0.0.0.0:18080
  python3 aqara-api-adapter.py --port 18088              # 自定义端口
  python3 aqara-api-adapter.py --backend http://10.0.0.1:8080/v1  # 自定义后端
  python3 aqara-api-adapter.py --port 18888 --log /tmp/adapter.log

工具配置:
  Claude Code CLI:  settings.json → env.ANTHROPIC_BASE_URL = "http://127.0.0.1:18080"
  Codex CLI:        config.toml → base_url = "http://127.0.0.1:18080", wire_api = "responses"
  Pi:               models.json → openai-completions provider, baseUrl = "http://127.0.0.1:18080"
  CC Switch:        proxy_config → ANTHROPIC_BASE_URL = "http://127.0.0.1:18080"
        """,
    )
    parser.add_argument(
        "--port", "-p", type=int, default=DEFAULT_PORT,
        help=f"监听端口 (默认: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--backend", "-b", default=DEFAULT_BACKEND,
        help=f"后端 API 网关地址 (默认: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--api-key", "-k", default=None,
        help="API Key (也可通过 AQARA_API_KEY 环境变量设置)",
    )
    parser.add_argument(
        "--log", "-l", default=None,
        help="日志文件路径 (默认: 仅终端输出)",
    )
    parser.add_argument(
        "--model-map-file", "-m", default=None,
        help="模型映射 JSON 文件 (可选, 格式: {\"claude-name\": \"aqara-name\"})",
    )

    args = parser.parse_args()

    # API Key
    api_key = args.api_key or os.environ.get("AQARA_API_KEY", "")
    if not api_key:
        print("❌ 错误: 未设置 API Key。请使用 --api-key 或 AQARA_API_KEY 环境变量。", file=sys.stderr)
        sys.exit(1)

    # 模型映射
    model_map = MODEL_MAP.copy()
    if args.model_map_file:
        try:
            with open(args.model_map_file) as f:
                model_map.update(json.load(f))
        except Exception as e:
            print(f"⚠️  读取模型映射文件失败: {e}，使用默认映射", file=sys.stderr)

    # 日志
    global log
    log = _make_logger(args.log)

    log("=" * 55)
    log("  Aqara AI API Adapter v2.0")
    log(f"  监听: 0.0.0.0:{args.port}")
    log(f"  后端: {args.backend}")
    log(f"  模型数: {len(AVAILABLE_MODELS)}")
    log("  路由: /v1/messages, /v1/responses, /v1/chat/completions")
    log("=" * 55)

    # 注入配置到 Handler
    Handler.api_key = api_key
    Handler.backend_url = args.backend
    Handler.model_map = model_map
    Handler.available_models = AVAILABLE_MODELS

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    log("✅ 适配器已启动，按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
