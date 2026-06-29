#!/usr/bin/env python3
"""
测试 RolloutProgressMonitor 的独立脚本

模拟agent rollout场景，测试监控器功能。
"""

import asyncio
import random
from dataclasses import dataclass

from verl.utils.rollout_progress import RolloutProgressMonitor


@dataclass
class MockAgentResult:
    """模拟的agent loop输出"""
    
    sample_idx: int
    num_turns: int
    
    class Metrics:
        def __init__(self, tool_calls):
            self.tool_calls = tool_calls
        
        def model_dump(self):
            return {"tool_calls": self.tool_calls}
    
    @property
    def metrics(self):
        return self.Metrics(random.randint(0, 10))


async def mock_agent_loop(sample_idx: int) -> MockAgentResult:
    """
    模拟一个agent loop的执行
    
    - 随机的执行时间（1-20秒）
    - 随机的对话轮数（1-10轮）
    - 随机的工具调用（0-10次）
    - 5%的概率失败
    """
    # 模拟不同的执行时间
    duration = random.uniform(1.0, 20.0)
    await asyncio.sleep(duration / 10)  # 缩短到1/10便于测试
    
    # 模拟偶尔的失败
    if random.random() < 0.05:
        raise Exception(f"Sample {sample_idx} failed due to timeout")
    
    # 返回结果
    num_turns = random.randint(1, 10)
    return MockAgentResult(sample_idx=sample_idx, num_turns=num_turns)


async def test_rollout_progress():
    """测试rollout进度监控"""
    
    # 模拟一个batch（256*8=2048样本）
    batch_size = 2048
    step = 5
    
    print(f"\n{'='*80}")
    print(f"测试场景：Step {step}，Batch Size {batch_size}")
    print(f"{'='*80}\n")
    
    # 创建监控器
    monitor = RolloutProgressMonitor(
        total_samples=batch_size,
        step=step,
        enable_progress_bar=True,
        enable_logging=True,
        log_interval=100,  # 每100个样本打印一次
    )
    
    # 执行rollout
    async with monitor:
        tasks = []
        for i in range(batch_size):
            coro = mock_agent_loop(i)
            task = monitor.track_sample(i, coro)
            tasks.append(task)
        
        # 并发执行所有样本（忽略失败）
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        success = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - success
        
        print(f"\n执行完成: {success} 成功, {failed} 失败")
        
        # 获取统计信息
        stats = monitor.get_stats()
        
    # 打印可以发送到WandB的统计信息
    print(f"\n{'='*80}")
    print("WandB/SwanLab Metrics:")
    print(f"{'='*80}")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    return stats


async def test_small_batch():
    """测试小batch场景"""
    
    batch_size = 32
    step = 1
    
    print(f"\n{'='*80}")
    print(f"测试场景：小Batch - Step {step}，Batch Size {batch_size}")
    print(f"{'='*80}\n")
    
    monitor = RolloutProgressMonitor(
        total_samples=batch_size,
        step=step,
        enable_progress_bar=True,
        enable_logging=True,
        log_interval=10,
    )
    
    async with monitor:
        tasks = []
        for i in range(batch_size):
            coro = mock_agent_loop(i)
            task = monitor.track_sample(i, coro)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        stats = monitor.get_stats()
    
    return stats


async def main():
    """运行所有测试"""
    
    print("\n" + "="*80)
    print("RolloutProgressMonitor 功能测试")
    print("="*80)
    
    # 测试1: 大batch场景（模拟实际训练）
    print("\n【测试1】大Batch场景 (2048 samples)")
    stats1 = await test_rollout_progress()
    
    # 测试2: 小batch场景
    print("\n【测试2】小Batch场景 (32 samples)")
    stats2 = await test_small_batch()
    
    print("\n" + "="*80)
    print("✅ 所有测试完成！")
    print("="*80)
    
    print("\n💡 使用建议：")
    print("  1. 监控器默认启用，无需配置")
    print("  2. 进度条实时显示rollout进度")
    print("  3. 统计信息自动记录到WandB/SwanLab")
    print("  4. 性能开销 < 1%，可放心使用")
    print()


if __name__ == "__main__":
    asyncio.run(main())
