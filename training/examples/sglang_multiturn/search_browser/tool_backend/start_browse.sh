#!/bin/bash
cd "$(dirname "$0")"
source .env
echo "✅ 环境变量已加载"
echo "🚀 启动 Browse 服务..."
python browse_service.py
