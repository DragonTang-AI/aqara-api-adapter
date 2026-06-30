#!/bin/bash
# Aqara API 适配器自动启动脚本
# 检查端口占用，避免重复启动

API_KEY="${AQARA_API_KEY:-sk-f12208bbc17b18dfb36e083c08e07ed258e37522841422f1a5a67a0a426fc910}"
ADAPTER=~/WorkBuddy/2026-05-18-task-8/aqara-api-adapter.py

start_if_dead() {
    PORT=$1
    NAME=$2
    if lsof -i :$PORT -P 2>/dev/null | grep -q LISTEN; then
        echo "[$(date)] $NAME (:${PORT}) 已运行，跳过"
    else
        echo "[$(date)] 启动 $NAME (:${PORT})..."
        python3 "$ADAPTER" --port "$PORT" --api-key "$API_KEY" --log "/tmp/.adapter-${PORT}.log" &
    fi
}

start_if_dead 18090 "Claude-Desktop-Adapter"
start_if_dead 18080 "Codex-Adapter"

echo "[$(date)] 启动完成"
