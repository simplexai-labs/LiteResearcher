# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Rollout Progress Monitor for Agent Loop Batch Processing

用于监控单个step内batch样本的rollout进度，特别适合agent多轮对话场景。
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from tqdm.asyncio import tqdm


@dataclass
class SampleProgress:
    """单个样本的进度信息"""
    
    sample_idx: int
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    current_turn: int = 0
    total_turns: int = 0
    tool_calls: int = 0
    error: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """计算已用时间（秒）"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self):
        """转换为字典用于日志"""
        return {
            "sample_idx": self.sample_idx,
            "status": self.status,
            "duration": round(self.duration, 2),
            "turns": self.total_turns,
            "tool_calls": self.tool_calls,
            "error": self.error,
        }


class RolloutProgressMonitor:
    """
    Rollout进度监控器
    
    功能：
    1. 实时显示batch内样本完成进度
    2. 追踪每个样本的状态（运行中/已完成/失败）
    3. 统计性能指标（完成时间、工具调用等）
    4. 提供进度条显示
    
    使用示例：
        monitor = RolloutProgressMonitor(total_samples=2048, step=5)
        async with monitor:
            tasks = []
            for i in range(len(batch)):
                task = monitor.track_sample(i, self._run_agent_loop(...))
                tasks.append(task)
            results = await asyncio.gather(*tasks)
    """
    
    def __init__(
        self,
        total_samples: int,
        step: int = 0,
        enable_progress_bar: bool = True,
        enable_logging: bool = True,
        log_interval: int = 10,  # 每N个样本完成打印一次统计
        worker_id: Optional[int] = None,  # Worker ID（用于分布式场景）
        total_workers: Optional[int] = None,  # 总worker数
        global_total_samples: Optional[int] = None,  # 全局总样本数
    ):
        self.total_samples = total_samples
        self.step = step
        self.enable_progress_bar = enable_progress_bar
        self.enable_logging = enable_logging
        self.log_interval = log_interval
        self.worker_id = worker_id
        self.total_workers = total_workers
        self.global_total_samples = global_total_samples or total_samples
        
        # 进度追踪
        self.samples: dict[int, SampleProgress] = {}
        self.completed_count = 0
        self.failed_count = 0
        
        # 性能统计
        self.start_time = None
        self.end_time = None
        
        # 进度条
        self.pbar = None
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        """启动监控"""
        self.start_time = time.time()
        if self.enable_progress_bar:
            # 构建描述信息
            if self.worker_id is not None:
                desc = f"[Step {self.step}] Worker {self.worker_id}/{self.total_workers} Rollout"
                desc += f" (Global: {self.global_total_samples} samples)"
            else:
                desc = f"[Step {self.step}] Agent Rollout"
            
            self.pbar = tqdm(
                total=self.total_samples,
                desc=desc,
                unit="sample",
                ncols=140,
                colour="green",
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """结束监控，打印汇总"""
        self.end_time = time.time()
        if self.pbar:
            self.pbar.close()
        
        if self.enable_logging:
            self._print_summary()
    
    async def track_sample(self, sample_idx: int, coro):
        """
        追踪单个样本的执行
        
        Args:
            sample_idx: 样本索引
            coro: 要执行的协程（agent loop）
        
        Returns:
            协程的返回结果
        """
        # 初始化样本进度
        progress = SampleProgress(sample_idx=sample_idx, status="running", start_time=time.time())
        async with self._lock:
            self.samples[sample_idx] = progress
        
        try:
            # 执行agent loop
            result = await coro
            
            # 更新完成状态
            async with self._lock:
                progress.status = "completed"
                progress.end_time = time.time()
                progress.total_turns = getattr(result, "num_turns", 0)
                
                # 从metrics中提取工具调用次数
                if hasattr(result, "metrics"):
                    metrics_dict = result.metrics.model_dump() if hasattr(result.metrics, "model_dump") else {}
                    progress.tool_calls = int(metrics_dict.get("tool_calls", 0))
                
                self.completed_count += 1
                
                # 更新进度条
                if self.pbar:
                    self.pbar.update(1)
                    # 更新描述信息
                    self.pbar.set_postfix({
                        "completed": self.completed_count,
                        "failed": self.failed_count,
                        "avg_time": f"{self._get_avg_duration():.1f}s",
                    })
                
                # 定期打印详细统计
                if self.enable_logging and self.completed_count % self.log_interval == 0:
                    self._print_progress()
            
            return result
            
        except Exception as e:
            # 记录失败
            async with self._lock:
                progress.status = "failed"
                progress.end_time = time.time()
                progress.error = str(e)
                self.failed_count += 1
                
                if self.pbar:
                    self.pbar.update(1)
                    self.pbar.set_postfix({
                        "completed": self.completed_count,
                        "failed": self.failed_count,
                        "avg_time": f"{self._get_avg_duration():.1f}s",
                    })
            
            raise
    
    def update_sample_progress(self, sample_idx: int, current_turn: int, tool_calls: int = None):
        """
        更新样本的中间进度（可选，用于更细粒度的追踪）
        
        Args:
            sample_idx: 样本索引
            current_turn: 当前对话轮数
            tool_calls: 工具调用次数
        """
        if sample_idx in self.samples:
            progress = self.samples[sample_idx]
            progress.current_turn = current_turn
            if tool_calls is not None:
                progress.tool_calls = tool_calls
    
    def _get_avg_duration(self) -> float:
        """计算已完成样本的平均时间"""
        completed_samples = [s for s in self.samples.values() if s.status == "completed"]
        if not completed_samples:
            return 0.0
        return sum(s.duration for s in completed_samples) / len(completed_samples)
    
    def _get_duration_stats(self) -> dict:
        """获取时间统计"""
        completed_samples = [s for s in self.samples.values() if s.status == "completed"]
        if not completed_samples:
            return {"min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0}
        
        durations = sorted([s.duration for s in completed_samples])
        n = len(durations)
        return {
            "min": durations[0],
            "max": durations[-1],
            "avg": sum(durations) / n,
            "p50": durations[n // 2],
            "p95": durations[int(n * 0.95)] if n > 1 else durations[-1],
        }
    
    def _print_progress(self):
        """打印进度统计"""
        stats = self._get_duration_stats()
        worker_info = f"Worker {self.worker_id}/{self.total_workers} " if self.worker_id is not None else ""
        print(f"\n[Step {self.step}] {worker_info}Rollout Progress: {self.completed_count}/{self.total_samples} completed")
        if self.worker_id is not None:
            print(f"  🌐 Global Progress: ~{self.completed_count * self.total_workers}/{self.global_total_samples} (estimated)")
        print(f"  ⏱️  Duration: avg={stats['avg']:.1f}s, p50={stats['p50']:.1f}s, p95={stats['p95']:.1f}s")
        print(f"  ✅ Success: {self.completed_count}, ❌ Failed: {self.failed_count}")
    
    def _print_summary(self):
        """打印最终汇总"""
        total_duration = self.end_time - self.start_time
        stats = self._get_duration_stats()
        
        # 统计对话轮数和工具调用
        completed_samples = [s for s in self.samples.values() if s.status == "completed"]
        total_turns = sum(s.total_turns for s in completed_samples)
        total_tool_calls = sum(s.tool_calls for s in completed_samples)
        avg_turns = total_turns / len(completed_samples) if completed_samples else 0
        avg_tool_calls = total_tool_calls / len(completed_samples) if completed_samples else 0
        
        worker_prefix = f"Worker {self.worker_id}/{self.total_workers} - " if self.worker_id is not None else ""
        
        print("\n" + "=" * 80)
        print(f"[Step {self.step}] {worker_prefix}Rollout Complete Summary")
        print("=" * 80)
        if self.worker_id is not None:
            print(f"🌐 Global Total:      {self.global_total_samples} samples across {self.total_workers} workers")
            print(f"📦 This Worker:       {self.total_samples} samples")
        else:
            print(f"📊 Total Samples:     {self.total_samples}")
        print(f"✅ Completed:         {self.completed_count} ({self.completed_count/self.total_samples*100:.1f}%)")
        print(f"❌ Failed:            {self.failed_count} ({self.failed_count/self.total_samples*100:.1f}%)")
        print(f"⏱️  Total Time:        {total_duration:.1f}s ({total_duration/60:.1f}min)")
        print(f"⚡ Throughput:        {self.total_samples/total_duration:.1f} samples/s")
        print(f"\n🔄 Agent Statistics:")
        print(f"   Avg Turns/Sample:  {avg_turns:.1f}")
        print(f"   Avg Tools/Sample:  {avg_tool_calls:.1f}")
        print(f"   Total Turns:       {total_turns}")
        print(f"   Total Tool Calls:  {total_tool_calls}")
        print(f"\n⏱️  Sample Duration:")
        print(f"   Min:  {stats['min']:.1f}s")
        print(f"   P50:  {stats['p50']:.1f}s")
        print(f"   P95:  {stats['p95']:.1f}s")
        print(f"   Max:  {stats['max']:.1f}s")
        print(f"   Avg:  {stats['avg']:.1f}s")
        
        # 最慢的5个样本
        slowest = sorted(completed_samples, key=lambda s: s.duration, reverse=True)[:5]
        if slowest:
            print(f"\n🐌 Slowest 5 Samples:")
            for i, s in enumerate(slowest, 1):
                print(f"   {i}. Sample {s.sample_idx}: {s.duration:.1f}s ({s.total_turns} turns, {s.tool_calls} tools)")
        
        print("=" * 80 + "\n")
    
    def get_stats(self) -> dict:
        """获取统计数据（用于记录到wandb等）"""
        stats = self._get_duration_stats()
        completed_samples = [s for s in self.samples.values() if s.status == "completed"]
        
        total_turns = sum(s.total_turns for s in completed_samples)
        total_tool_calls = sum(s.tool_calls for s in completed_samples)
        
        return {
            "rollout/total_samples": self.total_samples,
            "rollout/completed": self.completed_count,
            "rollout/failed": self.failed_count,
            "rollout/success_rate": self.completed_count / self.total_samples if self.total_samples > 0 else 0,
            "rollout/duration_min": stats["min"],
            "rollout/duration_max": stats["max"],
            "rollout/duration_avg": stats["avg"],
            "rollout/duration_p50": stats["p50"],
            "rollout/duration_p95": stats["p95"],
            "rollout/total_duration": self.end_time - self.start_time if self.end_time else 0,
            "rollout/throughput": self.total_samples / (self.end_time - self.start_time) if self.end_time and self.start_time else 0,
            "rollout/avg_turns_per_sample": total_turns / len(completed_samples) if completed_samples else 0,
            "rollout/avg_tools_per_sample": total_tool_calls / len(completed_samples) if completed_samples else 0,
            "rollout/total_turns": total_turns,
            "rollout/total_tool_calls": total_tool_calls,
        }
