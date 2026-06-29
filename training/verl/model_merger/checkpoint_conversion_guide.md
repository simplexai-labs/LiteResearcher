# FSDP Checkpoint 转换指南

本文档介绍如何将 VERL 训练保存的 FSDP checkpoint 转换为标准 HuggingFace 格式。

## 目录结构

训练保存的 checkpoint 结构如下：

```
global_step_18/
├── actor/
│   ├── model_world_size_8_rank_*.pt   # 8个分片的模型权重
│   ├── optim_world_size_8_rank_*.pt   # 8个分片的优化器状态
│   ├── extra_state_*.pt               # 额外状态（LR scheduler, RNG等）
│   ├── fsdp_config.json               # FSDP 配置
│   └── huggingface/                   # 仅包含配置和tokenizer (16MB)
│       ├── config.json
│       ├── tokenizer.json
│       └── ... (无模型权重)
└── data.pt                            # 训练状态
```

**注意**：`huggingface/` 目录默认只保存配置和 tokenizer，**不包含模型权重**。

---

## 方法一：使用 VERL 官方 model_merger（推荐）

### 基本用法

```bash
cd /share/project/wanli/Search_Agent/verl

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_nokl/global_step_18/actor \
    --target_dir checkpoints/hf_models/global_step_18
```

### 完整脚本示例

```bash
#!/bin/bash
# convert_fsdp_to_hf.sh

# 激活环境
source /share/project/wanli/env/verl-v060/bin/activate

SOURCE_CKPT="checkpoints/YOUR_PROJECT/global_step_18/actor"
TARGET_DIR="checkpoints/hf_models/global_step_18"

echo "🔄 转换 FSDP checkpoint 到 HuggingFace 格式..."
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$SOURCE_CKPT" \
    --target_dir "$TARGET_DIR"

echo "✅ 完成！模型保存在: $TARGET_DIR"
```

### 参数说明

| 参数 | 说明 | 必需 |
|------|------|------|
| `--backend` | 后端类型：`fsdp` 或 `megatron` | ✅ |
| `--local_dir` | FSDP checkpoint 的 `actor` 目录路径 | ✅ |
| `--target_dir` | 输出的 HuggingFace 模型目录 | ✅ |
| `--trust-remote-code` | 是否信任远程代码（某些模型需要） | ❌ |
| `--hf_upload_path` | HuggingFace Hub 仓库ID（如：`username/model-name`） | ❌ |
| `--private` | 上传为私有仓库 | ❌ |

### 验证转换结果

```bash
python -m verl.model_merger test \
    --backend fsdp \
    --local_dir checkpoints/YOUR_PROJECT/global_step_18/actor \
    --test_hf_dir checkpoints/hf_models/global_step_18
```

这会对比转换后的模型与原始配置，确保权重正确。

---

## 方法二：训练时自动保存 HF 格式

如果你想在训练时**自动保存** HuggingFace 格式的模型，在训练配置中添加：

```yaml
# 配置文件（如 config.yaml）
actor_rollout_ref:
  actor:
    checkpoint:
      save_hf_model: true  # 启用 HF 模型保存
```

或在命令行中：

```bash
python -m verl.trainer.main_ppo \
    +trainer=ppo_trainer.yaml \
    actor_rollout_ref.actor.checkpoint.save_hf_model=true \
    ...
```

这样每次保存 checkpoint 时，`huggingface/` 目录会包含完整的模型权重（`.safetensors` 格式）。

---

## 加载转换后的模型

### 使用 Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "path/to/hf_model",
    torch_dtype="auto",
    device_map="auto"
)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/hf_model")

# 推理
inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 使用 vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/hf_model \
    --tensor-parallel-size 8 \
    --port 8000
```

### 使用 SGLang

```bash
python -m sglang.launch_server \
    --model-path /path/to/hf_model \
    --tp 8 \
    --port 8000
```

---

## 方法三：转换 GPU 数量（8→16 GPU）

如果你需要在不同 GPU 数量的环境间迁移 checkpoint，使用：

```bash
bash scripts/convert_checkpoint_8to16.sh \
    /path/to/global_step_18 \
    /path/to/global_step_18_16gpu
```

这会将 8-GPU 的 checkpoint 转换为 16-GPU 格式。

---

## 常见问题

### Q1: 转换后模型很大（~17GB）？

A: 这是正常的。FSDP 将 8 个分片（每个 ~2.2GB）合并成完整模型。如果只需要单卡推理，可以考虑量化：

```python
model = AutoModelForCausalLM.from_pretrained(
    "path/to/hf_model",
    load_in_8bit=True,  # 或 load_in_4bit=True
    device_map="auto"
)
```

### Q2: 转换时出现 OOM（内存不足）？

A: 使用 CPU 模式（较慢）：

```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$SOURCE_CKPT" \
    --target_dir "$TARGET_DIR" \
    --use_cpu_initialization
```

### Q3: 如何只保留最新几个 checkpoint？

A: 在训练配置中设置：

```yaml
trainer:
  save_freq: 4              # 每 4 步保存一次
  max_ckpt_to_keep: 3       # 只保留最近 3 个 checkpoint
```

### Q4: `huggingface/` 目录只有 16MB，缺少模型权重？

A: 这是默认行为。要保存 HF 格式权重，需启用 `save_hf_model=true`（见方法二），或者使用 `model_merger` 手动转换（见方法一）。

---

## 相关文件

- [FSDP checkpoint manager](verl/utils/checkpoint/fsdp_checkpoint_manager.py) - FSDP checkpoint 保存/加载逻辑
- [Model merger](verl/model_merger/) - FSDP/Megatron 到 HuggingFace 的转换工具
- [Checkpoint conversion script](verl/utils/checkpoint/convert_fsdp_checkpoint.py) - GPU 数量转换脚本
- [Multi-GPU conversion script](scripts/convert_checkpoint_8to16.sh) - 8→16 GPU 转换封装

---

## 参考资料

- [VERL Checkpoint 文档](https://verl.readthedocs.io/en/latest/advance/checkpoint.html)
- [FSDP 官方文档](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_beginner_guide.html)
- [Transformers 文档](https://huggingface.co/docs/transformers/)
