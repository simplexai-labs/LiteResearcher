# qwen3-4B-RL · Selected Trajectories Viewer

可视化展示一组从 **qwen3-4B-RL** (stage2 onpolicy, ckpt220, bs128, all-rag, length 48k, global_step_420) 模型在 8 个公开 deep-research benchmark 上的真实 rollout 轨迹。

每个 case 都满足：
- ✅ `judge.correct == True`（模型答案与 reference 匹配）
- ✅ 经过 4 轮 Opus-4.7 (1M context) 独立 subagent 严格审查，**无 benchmark Q-A dump leak**
- ✅ 答案确实从检索证据严谨推导得出，**无 fabrication、无 hedged guess**
- ✅ 单轨迹普遍 40–170 步、20K–110K think 字符，展示 planner / verifier / multi-hop 行为

## 目录结构

```
qwen3-4b-rl-cases/
├── README.md          # 本文件
├── index.html         # viewer 入口
├── viewer.js          # 渲染逻辑（vanilla JS，无依赖）
└── cases/
    ├── index.json     # 所有 benchmark 的总索引
    ├── Frame.json
    ├── GAIA.json
    ├── HLE.json
    ├── Seal0.json
    ├── WebwalkerQA.json
    ├── Xbench.json
    ├── browsecomp-zh.json
    └── browsecomp.json
```

## 本地预览

viewer 必须通过 HTTP 服务器打开（浏览器不允许 `file://` 协议下 `fetch()` 本地 JSON）。最简单的方式：

```bash
cd qwen3-4b-rl-cases
python3 -m http.server 8000
# 然后浏览器访问 http://127.0.0.1:8000/
```

或者用 Node：

```bash
npx serve qwen3-4b-rl-cases -p 8000
```

## 部署到网站

直接把整个 `qwen3-4b-rl-cases/` 目录作为**静态资源**上传到任何能托管静态文件的地方：
- GitHub Pages：把目录内容 push 到 `gh-pages` 分支根目录（或 `docs/` 子目录）
- Cloudflare Pages / Vercel / Netlify：drag-and-drop 整个目录
- 自建 Nginx：把目录放到 `/var/www/<site>/` 下即可

确保 `index.html`、`viewer.js` 与 `cases/` 都在同一路径下，相对路径就能 work。无后端、无构建步骤、无外部 CDN 依赖（所有样式都内联在 `index.html`）。

## 可视化界面说明

- **左侧栏**：按 benchmark 折叠展开，点击 benchmark 名称可展开/收起；点击 case 行加载该轨迹。case 行显示 `id`、`score`（自动评分）、和问题前 80 字符。
- **右侧主区**：从上到下：
  - **问题** + reference / final answer 对比
  - 一系列 **step**：
    - 🧠 蓝色 = 模型 think（chain-of-thought）
    - 🔍 绿色 = 调用 `search` 工具（含具体 queries）
    - 🌐 橙色 = 调用 `visit` 工具（含 URL + goal）
    - 灰色 = 工具返回内容（默认折叠到 200px，可点 "展开" 看全文）
    - 紫色 = 模型最终 `<answer>` 块

## 选 case 清单

| benchmark | picks (id) | 亮点 |
|---|---|---|
| Frame | 680, 821 | 多跳推理 + 自我纠错 |
| GAIA | 56, 2 | 详尽枚举 + 精确算术 |
| HLE | 29, 315 | 音乐理论 IMSLP 验证 + 几何题从零坐标推导 |
| Seal0 | 110, 100 | M&A 表格直查 + 历史人物枚举 |
| WebwalkerQA | 158, 514 | 官方网站爬取 + ICSE 2002 PC 验证 |
| Xbench | 92, 63 | 奥运姐妹日期算术 + 文学→电影→奥运跨域链 |
| browsecomp-zh | 13, 53 | CNKI 论文每个 sub-clue 锚定 + 三 clue 全验证电影识别 |
| browsecomp | 844 | 6 个独立 biographical clue 多源 cited 验证 → Melissa Marr |

共 **15 个 case** 覆盖 **8 个 benchmark**。

## 数据 schema

`cases/<benchmark>.json` 形如：

```json
{
  "benchmark": "Frame",
  "source_folder": "...",
  "manual_picks": [680, 821],
  "n_total_correct": 682,
  "selected": 2,
  "cases": [
    {
      "id": 680,
      "benchmark": "Frame",
      "question": "...",
      "reference_answer": "...",
      "final_answer": "...",
      "judge_correct": true,
      "judge_verdict": "...",
      "score": 12.5,
      "stats": {
        "n_search": 36, "n_visit": 31, "n_domains": 2,
        "n_think_turns": 30, "total_think_chars": 108930,
        "domains": [...]
      },
      "source_file": "result_680_<timestamp>_rollout_1.json",
      "steps": [
        {"type": "question", "content": "..."},
        {"type": "assistant", "think": "...", "tool_calls": [
          {"name": "search", "queries": ["..."]} ,
          {"name": "visit", "urls": ["..."], "goal": "..."}
        ], "answer": null},
        {"type": "tool_response", "content": "..."},
        ...
        {"type": "assistant", "think": "...", "tool_calls": [], "answer": "<final answer>"}
      ]
    }
  ]
}
```

`cases/index.json` 包含每个 benchmark 的 picks summary 供 sidebar 使用。

## 数据来源

所有 rollout 来自 `qwen3-4B-RL` 自家训练 checkpoint：

```
stage2_onpolicy_new_ckpt220_bs128_all_rag-temp_1_length_48k/global_step_420/<benchmark>_pass@N
```

browsecomp (English) 单独来自同一 checkpoint 的 `_old` 目录（`global_step_420_old/browsecomp_400_pass@1`），因 `global_step_420` 未跑英文 browsecomp。

## 审查方法（如何避免 leak）

筛选脚本 `select_and_export.py`（未包含在本 zip 中，存于原仓库）维护了一份 `HACK_URL_PATTERNS` 黑名单，过滤掉访问过下列 benchmark Q-A dump 的 case：

- HuggingFace datasets / spaces: `callanwu/WebWalkerQA`, `vtllms/sealqa`, `OpenResearcher/web-bench`, `Kevin355/Who_and_When`, `inclusionAI/ASearcher-test-data`, `cais/hle`, `intelligent-internet/gaia-subset-benchmark`, `bstraehle/gaia`, `agents-course/Final_Assignment_Template` …
- GitHub: `openai/simple-evals`, `aymeric-roucher/GAIA`, `MinorJerry/WebVoyager`
- 论文 case study dump: `arxiv.org/.../2508.10874` (SSRL), `2510.25160`, `2504.21776`; `researchgate.net/publication/394488256`
- 通用兜底：路径含 `webwalkerqa/sealqa/browsecomp/asearcher/webvoyager/gaia_*/benchmark_gaia/hle-public/simple-evals/final_assignment_template` 的 HF 资源
- 剪贴板/匿名 paste: `pastebin.com/*`, `rentry.{co,org}/*`, `ghostbin.co/*`

每个最终 case 还经过 Opus 4.7 (1M context) subagent 的 4 轮独立审查（leak + 答案推导严谨度 + 完整逻辑链审计），确认 PR-grade。

## 许可

轨迹数据来自模型自身 rollout；展示形式（viewer.html / viewer.js）可自由使用、修改、再分发。