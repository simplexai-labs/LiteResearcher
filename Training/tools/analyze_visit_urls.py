#!/usr/bin/env python3
"""
📊 Visit Tool URL 分析脚本 (v2)
================================
统计 rollout trajectory 中 visit 工具调用的 URL：

核心逻辑：
  所有 visit URL → 排除「连接失败」→ 有效 URL 池
  有效 URL 池 → 查 RAG 索引 → 区分「真实 URL」vs「编造 URL」
  编造 URL → 区分「爬虫成功」vs「爬虫失败」

分类定义：
  - connect_fail: Browse 服务连接层面失败 (Browse error, ConnectionError, timed out, etc.)
  - browse_success: Browse 服务成功返回了网页摘要内容
  - browse_fail: Browse 服务收到请求但 cache miss / 无法处理 (could not be accessed, etc.)
  - no_response: tool_call 后没有对应的 tool_response

用法:
    python analyze_visit_urls.py \
        --input_dir  /path/to/rollout_trajectory/experiment_name \
        --output_dir /path/to/rollout_trajectory/experiment_name/visualization \
        --check_url  http://47.111.147.142:8010/batch_check_url
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============ HTTP Session ============

def create_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=5, pool_maxsize=20, max_retries=Retry(total=2, backoff_factor=0.5))
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def batch_check_urls(session: requests.Session, check_url: str, urls: list[str],
                     batch_size: int = 500, timeout: int = 60) -> dict[str, bool]:
    result = {}
    urls = [u for u in urls if isinstance(u, str)]
    unique_urls = list(set(urls))
    for i in range(0, len(unique_urls), batch_size):
        batch = unique_urls[i:i + batch_size]
        try:
            resp = session.post(check_url, json={"urls": batch}, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                results_map = data.get("results", {})
                for url in batch:
                    result[url] = results_map.get(url, False)
            else:
                for url in batch:
                    result[url] = False
        except Exception as e:
            print(f"  ⚠️  batch_check_url 失败: {e}")
            for url in batch:
                result[url] = False
    return result


# ============ 连接失败判定 ============

CONNECT_FAIL_MARKERS = [
    "Browse error",
    "ConnectionError",
    "ConnectionReset",
    "RemoteDisconnected",
    "timed out",
    "Max retries",
    "Connection aborted",
    "Connection refused",
]

BROWSE_FAIL_MARKERS = [
    "could not be accessed",
    "could not be processed",
    "no information is available",
]


def classify_response(resp_text: str) -> str:
    """分类 tool_response:
    Returns: 'connect_fail' | 'browse_fail' | 'browse_success'
    """
    if resp_text is None:
        return "no_response"
    # 连接失败优先判定
    for marker in CONNECT_FAIL_MARKERS:
        if marker in resp_text:
            return "connect_fail"
    # 爬虫/缓存失败
    for marker in BROWSE_FAIL_MARKERS:
        if marker in resp_text:
            return "browse_fail"
    return "browse_success"


# ============ 轨迹解析 ============

def extract_visit_urls_from_sample(output: str) -> list[tuple[str, str]]:
    """从单个样本提取所有 visit URL 及其分类
    Returns: [(url, category), ...] where category in {connect_fail, browse_fail, browse_success, no_response}
    """
    tc_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    tr_pattern = r'<tool_response>\s*(.*?)\s*</tool_response>'

    tool_calls = list(re.finditer(tc_pattern, output, re.DOTALL))
    tool_responses = list(re.finditer(tr_pattern, output, re.DOTALL))

    results = []
    for tc in tool_calls:
        try:
            parsed = json.loads(tc.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if parsed.get("name") != "visit":
            continue
        arguments = parsed.get("arguments", {})
        if not isinstance(arguments, dict):
            continue
        urls = arguments.get("url", [])
        if isinstance(urls, str):
            urls = [urls]
        urls = [u for u in urls if isinstance(u, str)]
        if not urls:
            continue

        tc_end = tc.end()
        resp_text = None
        for tr in tool_responses:
            if tr.start() > tc_end:
                resp_text = tr.group(1)
                break

        category = classify_response(resp_text)
        for u in urls:
            results.append((u, category))

    return results


def collect_step_data(input_dir: str) -> dict[int, str]:
    step_files = {}
    subdirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        for fname in os.listdir(subdir_path):
            if not fname.endswith(".jsonl"):
                continue
            try:
                step = int(fname.replace(".jsonl", ""))
            except ValueError:
                continue
            step_files[step] = os.path.join(subdir_path, fname)
    return step_files


def analyze_step(filepath: str) -> dict:
    """分析单个 step, 返回分类统计和 URL 列表"""
    all_url_cats = []  # [(url, category), ...]

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            output = data.get("output", "")
            if not output:
                continue
            all_url_cats.extend(extract_visit_urls_from_sample(output))

    # 统计
    stats = {
        "total_urls": len(all_url_cats),
        "connect_fail": 0,
        "browse_success": 0,
        "browse_fail": 0,
        "no_response": 0,
    }
    # 有效 URL（排除连接失败）
    valid_urls = []
    valid_url_cats = []

    for url, cat in all_url_cats:
        stats[cat] = stats.get(cat, 0) + 1
        if cat != "connect_fail" and cat != "no_response":
            valid_urls.append(url)
            valid_url_cats.append((url, cat))

    stats["valid_urls"] = len(valid_urls)
    return stats, valid_urls, valid_url_cats


# ============ 可视化 ============

def plot_results(steps, step_data_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    steps_arr = np.array(steps)

    # 提取各项数据
    total_urls = [d["total_urls"] for d in step_data_list]
    connect_fail = [d["connect_fail"] for d in step_data_list]
    valid_urls = [d["valid_urls"] for d in step_data_list]
    fabricated = [d["fabricated_count"] for d in step_data_list]
    real = [d["real_count"] for d in step_data_list]
    fab_ratio = [d["fabricated_ratio"] for d in step_data_list]
    fab_browse_success = [d["fabricated_browse_success"] for d in step_data_list]
    fab_browse_fail = [d["fabricated_browse_fail"] for d in step_data_list]

    # ============ 图1: 编造 URL 数量 + 比例 (双Y轴) ============
    fig, ax1 = plt.subplots(figsize=(16, 6))
    w = 0.35
    ax1.bar(steps_arr - w/2, real, width=w, label="Real URLs (in RAG)", color="#4CAF50", alpha=0.8)
    ax1.bar(steps_arr + w/2, fabricated, width=w, label="Fabricated URLs (not in RAG)", color="#F44336", alpha=0.8)
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("URL Count (excl. connect failures)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(steps_arr, [r * 100 for r in fab_ratio], "o-", color="#FF9800", linewidth=2, markersize=3,
             label="Fabricated Ratio (%)")
    ax2.set_ylabel("Fabricated Ratio (%)", fontsize=12, color="#FF9800")
    ax2.tick_params(axis="y", labelcolor="#FF9800")
    ax2.set_ylim(0, max(r * 100 for r in fab_ratio) * 1.3 if fab_ratio else 100)
    ax2.legend(loc="upper right", fontsize=9)

    plt.title("Fabricated URL Count & Ratio per Step\n(All visit URLs excl. connection failures)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(output_dir, "fabricated_url_count_and_ratio.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {p}")

    # ============ 图2: 编造 URL 中爬虫成功 vs 失败 ============
    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax1.bar(steps_arr - w/2, fab_browse_success, width=w,
            label="Fabricated + Browse Success", color="#FF9800", alpha=0.8)
    ax1.bar(steps_arr + w/2, fab_browse_fail, width=w,
            label="Fabricated + Browse Fail", color="#9C27B0", alpha=0.8)
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Fabricated URL Count", fontsize=12)
    ax1.legend(loc="upper left", fontsize=9)

    # 比例线
    ax2 = ax1.twinx()
    fab_success_ratio = [s / (s + f) * 100 if (s + f) > 0 else 0 for s, f in zip(fab_browse_success, fab_browse_fail)]
    ax2.plot(steps_arr, fab_success_ratio, "s-", color="#E91E63", linewidth=2, markersize=3,
             label="Fabricated Browse Success Rate (%)")
    ax2.set_ylabel("Browse Success Rate (%)", fontsize=12, color="#E91E63")
    ax2.tick_params(axis="y", labelcolor="#E91E63")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right", fontsize=9)

    plt.title("Fabricated URLs: Browse Success vs Fail\n(Among URLs not in RAG index)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(output_dir, "fabricated_url_browse_breakdown.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {p}")

    # ============ 图3: Fabricated Ratio 趋势 + 滑动平均 ============
    fig, ax = plt.subplots(figsize=(16, 5))
    ratios_pct = [r * 100 for r in fab_ratio]
    ax.plot(steps_arr, ratios_pct, "o-", color="#F44336", alpha=0.4, markersize=3, linewidth=1, label="Per-step")
    if len(ratios_pct) >= 5:
        window = min(10, max(1, len(ratios_pct) // 3))
        kernel = np.ones(window) / window
        smoothed = np.convolve(ratios_pct, kernel, mode="valid")
        offset = window // 2
        ax.plot(steps_arr[offset:offset + len(smoothed)], smoothed, "-", color="#D32F2F",
                linewidth=2.5, label=f"Moving avg (w={window})")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Fabricated URL Ratio (%)", fontsize=12)
    ax.set_ylim(0, max(ratios_pct) * 1.3 if ratios_pct else 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.title("Fabricated URL Ratio Trend\n(All visit URLs excl. connection failures)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(output_dir, "fabricated_url_ratio_trend.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {p}")

    # ============ 图4: 全景堆叠图 ============
    fig, ax = plt.subplots(figsize=(16, 6))
    # 从下到上: real_browse_success, real_browse_fail, fab_browse_success, fab_browse_fail, connect_fail
    real_bs = [d["real_browse_success"] for d in step_data_list]
    real_bf = [d["real_browse_fail"] for d in step_data_list]

    bottom = np.zeros(len(steps))
    ax.bar(steps_arr, real_bs, width=0.8, bottom=bottom, label="Real + Browse Success", color="#4CAF50", alpha=0.8)
    bottom += np.array(real_bs)
    ax.bar(steps_arr, real_bf, width=0.8, bottom=bottom, label="Real + Browse Fail", color="#8BC34A", alpha=0.6)
    bottom += np.array(real_bf)
    ax.bar(steps_arr, fab_browse_success, width=0.8, bottom=bottom, label="Fabricated + Browse Success", color="#FF9800", alpha=0.8)
    bottom += np.array(fab_browse_success)
    ax.bar(steps_arr, fab_browse_fail, width=0.8, bottom=bottom, label="Fabricated + Browse Fail", color="#9C27B0", alpha=0.7)
    bottom += np.array(fab_browse_fail)
    ax.bar(steps_arr, connect_fail, width=0.8, bottom=bottom, label="Connection Fail", color="#607D8B", alpha=0.5)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("URL Count", fontsize=12)
    ax.legend(loc="upper left", fontsize=8)
    plt.title("Full URL Breakdown per Step (Stacked)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(output_dir, "url_full_breakdown_stacked.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {p}")

    # ============ 图5: 总结 ============
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 5a: 有效 URL 池中的饼图
    total_fab = sum(fabricated)
    total_real = sum(real)
    if total_fab + total_real > 0:
        axes[0].pie([total_real, total_fab],
                     labels=[f"Real\n({total_real:,})", f"Fabricated\n({total_fab:,})"],
                     colors=["#4CAF50", "#F44336"], autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
    axes[0].set_title("Valid URL Pool\n(excl. connect fail)", fontsize=12)

    # 5b: 编造 URL 中的爬虫结果饼图
    total_fab_bs = sum(fab_browse_success)
    total_fab_bf = sum(fab_browse_fail)
    if total_fab_bs + total_fab_bf > 0:
        axes[1].pie([total_fab_bs, total_fab_bf],
                     labels=[f"Browse OK\n({total_fab_bs:,})", f"Browse Fail\n({total_fab_bf:,})"],
                     colors=["#FF9800", "#9C27B0"], autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
    axes[1].set_title("Fabricated URLs\nBrowse Outcome", fontsize=12)

    # 5c: 文字摘要
    axes[2].axis("off")
    total_all = sum(total_urls)
    total_cf = sum(connect_fail)
    total_valid = sum(valid_urls)
    stats_text = (
        f"Total Steps: {len(steps)}\n"
        f"Step Range: {min(steps)} - {max(steps)}\n"
        f"{'─' * 32}\n"
        f"Total Visit URLs:      {total_all:>8,}\n"
        f"  Connection Fail:     {total_cf:>8,}  ({total_cf/total_all*100:.1f}%)\n"
        f"  Valid URL Pool:      {total_valid:>8,}  ({total_valid/total_all*100:.1f}%)\n"
        f"{'─' * 32}\n"
        f"Valid Pool Breakdown:\n"
        f"  Real URLs:           {total_real:>8,}  ({total_real/total_valid*100:.1f}%)\n"
        f"  Fabricated URLs:     {total_fab:>8,}  ({total_fab/total_valid*100:.1f}%)\n"
        f"{'─' * 32}\n"
        f"Fabricated URL Outcome:\n"
        f"  Browse Success:      {total_fab_bs:>8,}  ({total_fab_bs/total_fab*100:.1f}%)\n"
        f"  Browse Fail:         {total_fab_bf:>8,}  ({total_fab_bf/total_fab*100:.1f}%)\n"
    ) if steps else "No data"
    axes[2].text(0.05, 0.5, stats_text, transform=axes[2].transAxes, fontsize=10.5,
                 verticalalignment="center", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    axes[2].set_title("Summary Statistics", fontsize=12)

    fig.suptitle("Visit Tool URL Analysis Summary (v2)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(output_dir, "fabricated_url_summary.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {p}")


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="分析 visit 工具的 fabricated URL 统计 (v2)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--check_url", type=str, default="http://47.111.147.142:8010/batch_check_url")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    print("=" * 70)
    print("📊 Visit Tool URL Analysis (v2)")
    print("=" * 70)
    print(f"  输入: {args.input_dir}")
    print(f"  输出: {args.output_dir}")
    print(f"  RAG:  {args.check_url}")
    print()

    # 1. 收集 step 文件
    print("📂 扫描 step 文件...")
    step_files = collect_step_data(args.input_dir)
    if not step_files:
        print("  ❌ 未找到 step 文件")
        sys.exit(1)
    sorted_steps = sorted(step_files.keys())
    print(f"  找到 {len(sorted_steps)} 个 step: {sorted_steps[0]} - {sorted_steps[-1]}")
    print()

    # 2. HTTP session
    session = create_session()
    print("🔗 测试 RAG 连接...")
    try:
        resp = session.post(args.check_url, json={"urls": ["https://test.example.com"]}, timeout=10)
        print(f"  ✅ OK (status={resp.status_code})")
    except Exception as e:
        print(f"  ⚠️  失败: {e}")
    print()

    # 3. 逐 step 分析
    print("🔍 逐 step 分析...")
    all_step_data = []

    for i, step in enumerate(sorted_steps):
        filepath = step_files[step]
        stats, valid_urls, valid_url_cats = analyze_step(filepath)

        # 查 RAG
        url_exists = {}
        if valid_urls:
            url_exists = batch_check_urls(session, args.check_url, valid_urls,
                                          batch_size=args.batch_size, timeout=args.timeout)

        # 分类统计
        fabricated_count = 0
        real_count = 0
        fabricated_browse_success = 0
        fabricated_browse_fail = 0
        real_browse_success = 0
        real_browse_fail = 0

        for url, cat in valid_url_cats:
            exists = url_exists.get(url, False)
            if exists:
                real_count += 1
                if cat == "browse_success":
                    real_browse_success += 1
                else:
                    real_browse_fail += 1
            else:
                fabricated_count += 1
                if cat == "browse_success":
                    fabricated_browse_success += 1
                else:
                    fabricated_browse_fail += 1

        valid_total = stats["valid_urls"]
        fab_ratio = fabricated_count / valid_total if valid_total > 0 else 0.0

        step_result = {
            **stats,
            "step": step,
            "fabricated_count": fabricated_count,
            "real_count": real_count,
            "fabricated_ratio": fab_ratio,
            "fabricated_browse_success": fabricated_browse_success,
            "fabricated_browse_fail": fabricated_browse_fail,
            "real_browse_success": real_browse_success,
            "real_browse_fail": real_browse_fail,
        }
        all_step_data.append(step_result)

        if (i + 1) % 10 == 0 or i == 0 or i == len(sorted_steps) - 1:
            print(f"  Step {step:>4d}: total={stats['total_urls']:>5d} conn_fail={stats['connect_fail']:>4d} "
                  f"valid={valid_total:>5d} fab={fabricated_count:>4d}({fab_ratio:.1%}) "
                  f"[fab_ok={fabricated_browse_success} fab_fail={fabricated_browse_fail}]  "
                  f"[{i+1}/{len(sorted_steps)}]")

    print()

    # 4. 保存详细数据
    os.makedirs(args.output_dir, exist_ok=True)
    detail_path = os.path.join(args.output_dir, "visit_url_analysis.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for item in all_step_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"💾 详细数据: {detail_path}")

    # 5. 图表
    print()
    print("📈 生成图表...")
    plot_results(sorted_steps, all_step_data, args.output_dir)

    # 6. 总结
    total_all = sum(d["total_urls"] for d in all_step_data)
    total_cf = sum(d["connect_fail"] for d in all_step_data)
    total_valid = sum(d["valid_urls"] for d in all_step_data)
    total_fab = sum(d["fabricated_count"] for d in all_step_data)
    total_real = sum(d["real_count"] for d in all_step_data)
    total_fab_bs = sum(d["fabricated_browse_success"] for d in all_step_data)
    total_fab_bf = sum(d["fabricated_browse_fail"] for d in all_step_data)

    print()
    print("=" * 70)
    print("📊 总结")
    print("=" * 70)
    print(f"  总 Step 数:           {len(sorted_steps)}")
    print(f"  Visit URL 总数:       {total_all:,}")
    print(f"    连接失败:           {total_cf:,} ({total_cf/total_all*100:.1f}%)")
    print(f"    有效 URL 池:        {total_valid:,} ({total_valid/total_all*100:.1f}%)")
    print(f"  ────────────────────────────")
    print(f"  有效池中:")
    print(f"    Real URL:           {total_real:,} ({total_real/total_valid*100:.1f}%)")
    print(f"    Fabricated URL:     {total_fab:,} ({total_fab/total_valid*100:.1f}%)")
    print(f"  ────────────────────────────")
    print(f"  编造 URL 中:")
    print(f"    爬虫成功:           {total_fab_bs:,} ({total_fab_bs/total_fab*100:.1f}%)")
    print(f"    爬虫失败:           {total_fab_bf:,} ({total_fab_bf/total_fab*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
