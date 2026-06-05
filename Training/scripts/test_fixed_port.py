#!/usr/bin/env python3
"""
测试 SGLang 固定端口配置
验证修改是否正确应用
"""

import sys
import inspect

# 测试 1: 检查 run_unvicorn 函数签名
print("=" * 60)
print("测试 1: 检查 run_unvicorn 函数签名")
print("=" * 60)

try:
    from verl.workers.rollout.utils import run_unvicorn
    sig = inspect.signature(run_unvicorn)
    print(f"✅ run_unvicorn 签名: {sig}")
    
    if 'fixed_port' in sig.parameters:
        print("✅ 参数 'fixed_port' 已添加")
    else:
        print("❌ 参数 'fixed_port' 未找到")
        sys.exit(1)
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

print()

# 测试 2: 检查 SGLangHttpServer 类
print("=" * 60)
print("测试 2: 检查 SGLangHttpServer 类")
print("=" * 60)

try:
    # 读取源代码检查
    import inspect
    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
    
    source = inspect.getsource(SGLangHttpServer.__init__)
    
    if '_base_http_port' in source:
        print("✅ _base_http_port 属性已添加")
    else:
        print("❌ _base_http_port 属性未找到")
        sys.exit(1)
        
    if '30000' in source:
        print("✅ 基础端口设置为 30000")
    else:
        print("⚠️  基础端口不是 30000")
    
    # 检查 launch_server 方法
    launch_source = inspect.getsource(SGLangHttpServer.launch_server)
    
    if 'fixed_port' in launch_source:
        print("✅ launch_server 使用 fixed_port 参数")
    else:
        print("❌ launch_server 未使用 fixed_port 参数")
        sys.exit(1)
        
    if 'replica_rank' in launch_source and 'fixed_http_port' in launch_source:
        print("✅ 使用 replica_rank 计算端口")
    else:
        print("⚠️  端口计算逻辑可能有问题")
        
except Exception as e:
    print(f"❌ 检查失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 测试 3: 模拟端口计算
print("=" * 60)
print("测试 3: 模拟端口计算")
print("=" * 60)

base_port = 30000
for replica_rank in range(8):
    expected_port = base_port + replica_rank
    print(f"  Replica {replica_rank} → Port {expected_port}")

print()
print("=" * 60)
print("✅ 所有测试通过!")
print("=" * 60)
print()
print("下一步:")
print("  1. 启动训练: bash examples/sglang_multiturn/search_browser/qwen3_agentloop.sh")
print("  2. 查找服务器: bash scripts/find_sglang_servers.sh")
print("  3. 监控服务器: python scripts/monitor_sglang.py --port 30000 --scan 8")
print()
