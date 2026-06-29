# GRPO_verl: 完整训练流程与模式切换机制

本文档详细说明 verl 框架中 GRPO (Group Relative Policy Optimization) 训练的完整流程，包括资源分配、数据流向、显存占用、FSDP 与 Rollout 的模式切换机制以及参数同步的具体实现。

**重点**：本文档特别详细描述了 **更新阶段（Update Stage）** 中 FSDP 如何计算梯度，哪些 tensor 参与反向传播。

---

## 快速概览：GRPO 更新阶段的核心要点

### 哪些 Tensor 参与反向传播？

| Tensor | 需要梯度？ | 来源 | 用途 |
|--------|-----------|------|------|
| **log_prob** | **✅ 是** | **FSDP Forward 计算** | **当前策略的 log probabilities**<br>→ 通过反向传播更新模型参数 |
| old_log_probs | ❌ 否 | compute_log_prob() 阶段 | 旧策略的 log probabilities<br>→ 用于计算 importance ratio |
| advantages | ❌ 否 | compute_advantage() 阶段（GRPO 公式） | 优势估计<br>→ 作为权重，不参与梯度计算 |
| ref_log_prob | ❌ 否 | compute_log_prob() 阶段（ref 模型） | 参考策略的 log probabilities<br>→ 用于 KL penalty（可选） |

### FSDP 梯度计算流程（ZeRO-3）

```
1. Forward Pass（verl/workers/actor/dp_actor.py:426-428）
   ├── FSDP 模型 forward → 计算 log_prob（需要梯度）
   ├── All-Gather 聚合完整参数（临时 16GB）
   ├── 计算 logits，提取 log_prob
   └── 释放临时参数（保留激活值用于 backward）

2. Loss 计算（verl/workers/actor/dp_actor.py:452-478）
   ├── Policy Loss = f(log_prob, old_log_probs, advantages)
   ├── 可选：Entropy Loss（使用 entropy，需要梯度）
   ├── 可选：KL Loss（使用 log_prob 和 ref_log_prob）
   └── 最终 loss 是标量，需要梯度

3. Backward Pass（verl/workers/actor/dp_actor.py:487）
   ├── loss.backward() ← 触发 FSDP 自动梯度计算
   ├── FSDP 逐层 All-Gather 完整参数（临时）
   ├── 计算梯度：∂loss/∂log_prob → ∂loss/∂W
   ├── Reduce-Scatter 梯度分片（每个 GPU 保留自己的分片）
   └── 释放临时参数

4. Optimizer Step（verl/workers/actor/dp_actor.py:499）
   ├── 梯度裁剪（All-Reduce 计算全局梯度范数）
   ├── AdamW 更新参数（每个 GPU 只更新自己的分片）
   └── 清空梯度
```

### 显存峰值（单个 Micro-Batch）

```
- Forward 峰值: ~22GB（临时聚合完整参数 16GB + 激活值 4GB + 分片 2GB）
- Backward 峰值: ~10.5GB（单层临时参数 0.5GB + 激活值 4GB + 分片 6GB）
- 稳定状态: ~6GB（分片 2GB + 优化器 4GB）
```

**详细内容见第 4.6 节。**

---

## 目录

1. [整体架构：Hybrid Engine 模式](#1-整体架构hybrid-engine-模式)
2. [资源初始化阶段](#2-资源初始化阶段)
3. [模型加载与显存占用](#3-模型加载与显存占用)
4. [PPO 完整训练迭代流程](#4-ppo-完整训练迭代流程)
   - **[4.6 更新阶段详解（重点）](#46-阶段-5更新-actor-模型fsdp-训练)**
5. [模式切换的详细机制](#5-模式切换的详细机制)
6. [参数同步的底层实现](#6-参数同步的底层实现)
7. [显存管理时间线](#7-显存管理时间线)
8. [代码位置索引](#8-代码位置索引)
9. [性能优化建议](#9-性能优化建议)

---

## 1. 整体架构：Hybrid Engine 模式

### 1.1 为什么需要 Hybrid Engine？

PPO 训练需要两个模型：
- **Actor FSDP 模型**：用于训练，需要梯度计算和参数更新（分片存储）
- **Rollout 模型**：用于推理生成，需要高效的 KV Cache 管理（完整模型）

如果分别部署，需要 **2倍的 GPU 资源**。Hybrid Engine 将两者共置（colocation）在同一组 GPU 上，节省 50% 资源。

### 1.2 Hybrid Engine 架构

```
每个 GPU 上的模型布局（以 Qwen3-8B 为例）:

┌─────────────────────────────────────────────────────────────┐
│ GPU 0 (80GB)                                                │
├─────────────────────────────────────────────────────────────┤
│ Actor FSDP 模型（训练用）                                   │
│   ├── 参数分片 [0:1B]: 2GB (可 offload 到 CPU)             │
│   ├── 优化器状态 (AdamW): 4GB (可 offload 到 CPU)          │
│   └── 梯度缓冲: 2GB                                         │
│                                                             │
│ Rollout 模型（推理用，vLLM/SGLang）                         │
│   ├── 完整参数 [0:8B]: 16GB (固定在 GPU)                   │
│   └── KV Cache: 32GB (动态分配/释放)                        │
│                                                             │
│ 总显存占用: 18-62GB (取决于当前模式)                        │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 两种运行模式

| 模式 | Actor FSDP | Rollout 模型 | Rollout KV Cache | 典型显存 |
|------|-----------|-------------|-----------------|---------|
| **Trainer Mode** | 在 GPU (2-8GB) | 在 GPU (16GB) | 已释放 (0GB) | ~22GB |
| **Rollout Mode** | 可 offload (0-2GB) | 在 GPU (16GB) | 已分配 (32GB) | ~50GB |

**核心思想**：通过动态切换模式，复用显存空间。

---

## 2. 资源初始化阶段

### 2.1 创建资源池和 Ray Actors

**代码位置**：
- `verl/trainer/main_ppo.py:166-189` - 创建资源池
- `verl/trainer/ppo/ray_trainer.py:661-796` - 初始化 workers

```python
# ========== 步骤 1: 创建资源池 ==========
# 文件: verl/trainer/main_ppo.py:166-189
resource_pool = RayResourcePool(
    process_on_nodes=[8],  # 8 个进程（每个 GPU 一个）
    use_gpu=True,
)

# 创建 Placement Groups（GPU 资源分配）
pgs = resource_pool.get_placement_groups()
# 结果: 8 个 Placement Groups，每个包含 1 个 GPU bundle


# ========== 步骤 2: 定义共置 Worker 配置 ==========
# 文件: verl/trainer/ppo/ray_trainer.py:745-766
class_dict = {
    "actor_rollout": RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        args=(...),
        kwargs={
            "config": actor_config,
            "role": "actor_rollout",  # Actor + Rollout 功能
        }
    ),
}


# ========== 步骤 3: 创建融合 Worker 类（WorkerDict）==========
# 文件: verl/single_controller/ray/base.py:749-790
worker_dict_cls = create_colocated_worker_cls(class_dict)

# WorkerDict 内部结构：
class WorkerDict(Worker):
    def __init__(self):
        self.worker_dict = {
            "actor_rollout": ActorRolloutRefWorker(role="actor_rollout"),
        }

    # 自动绑定方法（带前缀）
    def actor_rollout_generate_sequences(self, data):
        return self.worker_dict["actor_rollout"].generate_sequences(data)

    def actor_rollout_compute_log_prob(self, data):
        return self.worker_dict["actor_rollout"].compute_log_prob(data)

    def actor_rollout_update_actor(self, data):
        return self.worker_dict["actor_rollout"].update_actor(data)


# ========== 步骤 4: 创建 RayWorkerGroup（启动 8 个 Ray Actors）==========
# 文件: verl/single_controller/ray/base.py:361-444
wg_dict = RayWorkerGroup(
    resource_pool=resource_pool,
    ray_cls_with_init=worker_dict_cls,
)

# 此时创建了 8 个 Ray Actors：
# GPU 0: Ray Actor 0 (WorkerDict 实例)
# GPU 1: Ray Actor 1 (WorkerDict 实例)
# ...
# GPU 7: Ray Actor 7 (WorkerDict 实例)


# ========== 步骤 5: Spawn 创建 WorkerGroup 视图 ==========
# 文件: verl/single_controller/ray/base.py:478-512
spawn_wg = wg_dict.spawn(prefix_set={"actor_rollout"})

# spawn_wg = {
#     "actor_rollout": RayWorkerGroup(
#         _workers=[Actor 0, ..., Actor 7],  # 共享相同的 8 个 Ray Actors
#         方法: generate_sequences, compute_log_prob, update_actor  # 去掉前缀
#     )
# }


# ========== 步骤 6: 提取最终的 WorkerGroup ==========
# 文件: verl/trainer/ppo/ray_trainer.py:763
self.actor_rollout_wg = spawn_wg["actor_rollout"]
```

### 2.2 Spawn 机制详解

**Spawn 的核心作用**：实现共置（Colocation）

```python
# 物理结构（8 个 Ray Actors）：
┌─────────────────────────────────────────────────────────────┐
│ GPU 0: Ray Actor 0 (WorkerDict)                             │
│   ├── worker_dict["actor_rollout"] = ActorRolloutRefWorker │
│   └── worker_dict["ref"] = RefWorker (如果配置了)           │
├─────────────────────────────────────────────────────────────┤
│ GPU 1: Ray Actor 1 (WorkerDict)                             │
│   ├── worker_dict["actor_rollout"] = ActorRolloutRefWorker │
│   └── worker_dict["ref"] = RefWorker                        │
├─────────────────────────────────────────────────────────────┤
│ ... (GPU 2-6)                                               │
├─────────────────────────────────────────────────────────────┤
│ GPU 7: Ray Actor 7 (WorkerDict)                             │
│   ├── worker_dict["actor_rollout"] = ActorRolloutRefWorker │
│   └── worker_dict["ref"] = RefWorker                        │
└─────────────────────────────────────────────────────────────┘

# 逻辑结构（3 个 RayWorkerGroup 对象）：

wg_dict (原始)
├── _workers = [Actor 0, Actor 1, ..., Actor 7]
├── 方法: actor_rollout_generate_sequences, ref_compute_ref_log_prob
└── 用途: 初始化，通常不直接使用

spawn_wg["actor_rollout"] (视图1)
├── _workers = [Actor 0, Actor 1, ..., Actor 7]  # 共享相同的 Actors！
├── 方法: generate_sequences, update_actor  # 只暴露 actor_rollout 的方法
└── 调用时路由到 worker_dict["actor_rollout"]

spawn_wg["ref"] (视图2, 如果有)
├── _workers = [Actor 0, Actor 1, ..., Actor 7]  # 共享相同的 Actors！
├── 方法: compute_ref_log_prob  # 只暴露 ref 的方法
└── 调用时路由到 worker_dict["ref"]
```

**关键点**：
- 物理上只有 **8 个 Ray Actors**（8 个进程）
- 逻辑上有多个 RayWorkerGroup 对象（视图）
- 所有视图都引用相同的 8 个 Ray Actors
- Spawn 只是创建了不同的"视图"，不创建新的 Ray Actors
- **目的**：避免为每个功能创建独立的 GPU 进程，节省资源

---

## 3. 模型加载与显存占用

### 3.1 `__init__` - Worker 初始化

**代码位置**：`verl/workers/fsdp_workers.py:139-263`

在 Ray Actor 创建时，首先调用 `ActorRolloutRefWorker.__init__()`：

```python
# 文件: verl/workers/fsdp_workers.py:139-263

class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)
        self.config = config

        # ========== 步骤 1: 初始化分布式环境 ==========
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
            )

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # ========== 步骤 2: 创建 Device Mesh（分布式拓扑）==========
        from torch.distributed.device_mesh import init_device_mesh

        # Device Mesh 用于 FSDP/TP 分片
        # 例如：8 个 GPU → DeviceMesh([0, 1, 2, 3, 4, 5, 6, 7])
        self.device_mesh = init_device_mesh(
            get_device_name(),
            mesh_shape=(self.world_size,),
            mesh_dim_names=("fsdp",)
        )

        # ========== 步骤 3: 解析 role（决定创建哪些模型）==========
        # role 可能的值：
        # - "actor_rollout": 创建 Actor FSDP + Rollout 模型（Hybrid Engine）
        # - "actor": 仅创建 Actor FSDP
        # - "rollout": 仅创建 Rollout 模型
        # - "ref": 创建 Reference 模型

        self._is_actor = "actor" in role  # 是否包含 Actor 功能
        self._is_rollout = "rollout" in role  # 是否包含 Rollout 功能
        self._is_ref = "ref" in role  # 是否包含 Reference 功能

        # ========== 步骤 4: 配置 Offload 策略 ==========
        self._is_offload_param = config.actor.model.get("enable_parameter_offload", False)
        self._is_offload_optimizer = config.actor.optimizer.get("enable_optimizer_offload", False)

        # 注意：__init__ 阶段只是配置，不加载模型！
        # 实际的模型加载在 init_model() 中进行（由 Ray Trainer 调用）
```

**关键点**：
- `__init__` **不加载模型**，只初始化分布式环境和配置
- 模型加载在 `init_model()` 中（稍后调用）
- 此时显存占用：~0GB（仅分布式初始化开销 ~100MB）

---

### 3.2 `init_model()` - 模型初始化入口

**代码位置**：`verl/workers/fsdp_workers.py:760-820`

Ray Trainer 调用 `init_model()` 后，才真正加载模型：

```python
# 文件: verl/workers/fsdp_workers.py:760-820

@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def init_model(self):
    """初始化 Actor/Rollout/Ref 三个模型（FSDP 训练核心）"""

    # ========== 步骤 1: 加载 Actor FSDP 模型（用于训练）==========
    if self._is_actor:
        self.actor_module_fsdp, self.actor_optimizer, ... = self._build_model_optimizer(
            model_path=self.config.actor.model.path,  # 如 "Qwen/Qwen3-8B"
            fsdp_config=self.config.actor,
            role="actor",
        )
        # ↓ 详细流程见 3.3 节

    # ========== 步骤 2: 加载 Rollout 模型（用于推理）==========
    if self._is_rollout:
        self.rollout = self._build_rollout()
        # ↓ 详细流程见 3.4 节

    # ========== 步骤 3: 加载 Reference 模型（如果需要）==========
    if self._is_ref:
        self.ref_module_fsdp = self._build_model_optimizer(
            model_path=self.config.ref.model.path,
            fsdp_config=self.config.ref,
            role="ref",
        )

    # ========== 步骤 4: 配置 FSDP state_dict 类型 ==========
    # 文件: verl/workers/fsdp_workers.py:634-644
    if torch.distributed.get_world_size() == 1:
        FSDP.set_state_dict_type(
            self.actor_module_fsdp,
            state_dict_type=StateDictType.FULL_STATE_DICT,
        )
    else:
        FSDP.set_state_dict_type(
            self.actor_module_fsdp,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,  # ← 多 GPU 使用分片
        )

    # ========== 步骤 5: 切换到 Trainer 模式（初始状态）==========
    # 文件: verl/workers/fsdp_workers.py:650-656
    if rollout_config.mode == "sync" and self._is_actor:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.trainer_mode())
        # ↓ 切换到 Trainer 模式（释放 KV Cache）
```

---

### 3.3 `_build_model_optimizer()` - Actor FSDP 模型加载

**代码位置**：`verl/workers/fsdp_workers.py:268-589`

**核心问题**：显存中是否同时存在两份模型权重？

**答案**：**是的**！在 FSDP 初始化过程中，显存中短暂存在两份完整权重：

```python
# 文件: verl/workers/fsdp_workers.py:268-589

def _build_model_optimizer(self, model_path, fsdp_config, role="actor"):
    """构建 FSDP 模型和优化器"""

    # ========== 步骤 1: 加载 Tokenizer ==========
    # 文件: verl/workers/fsdp_workers.py:304-311
    self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
    self.processor = hf_processor(local_path, trust_remote_code=True)

    # ========== 步骤 2: 确定模型 dtype ==========
    # 文件: verl/workers/fsdp_workers.py:314-320
    torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
    # Actor: fp32（避免优化器在 bf16，影响训练稳定性）
    # Ref: bf16（节省显存，推理不需要高精度）

    # ========== 步骤 3: 加载模型配置 ==========
    # 文件: verl/workers/fsdp_workers.py:323-346
    actor_model_config = AutoConfig.from_pretrained(
        local_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2"
    )

    # ========== 步骤 4: 初始化模型（使用 meta tensor 或直接初始化）==========
    # 文件: verl/workers/fsdp_workers.py:348-389
    init_context = get_init_weight_context_manager(
        use_meta_tensor=not actor_model_config.tie_word_embeddings,
        mesh=self.device_mesh
    )
    # Meta tensor: 延迟初始化（不占用显存）
    # 如果 tie_word_embeddings=True，则直接初始化（占用显存）

    with init_context():
        # 加载预训练模型（从磁盘读取权重）
        actor_module = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=local_path,
            torch_dtype=torch_dtype,  # fp32 (Actor) 或 bf16 (Ref)
            config=actor_model_config,
            trust_remote_code=trust_remote_code,
        )

    # ========== 关键时刻：显存中有完整的模型权重 ==========
    # 此时每个 GPU 都加载了完整的 8B 参数（32GB，fp32 格式）
    # 显存占用：32GB / GPU

    # ========== 步骤 5: 应用优化（Liger/Monkey Patch/Gradient Checkpointing/LoRA）==========
    # 文件: verl/workers/fsdp_workers.py:392-432
    if use_liger:
        _apply_liger_kernel_to_instance(model=actor_module)  # 优化内核

    apply_monkey_patch(  # 注入自定义 forward
        model=actor_module,
        use_remove_padding=use_remove_padding,
        ulysses_sp_size=self.ulysses_sequence_parallel_size,
    )

    if enable_gradient_checkpointing:
        actor_module.gradient_checkpointing_enable()  # 省显存（recompute）

    if self._is_lora:
        actor_module = get_peft_model(actor_module, LoraConfig(...))  # LoRA 适配

    # ========== 步骤 6: FSDP 初始化（关键！）==========
    # 文件: verl/workers/fsdp_workers.py:447-540
    torch.distributed.barrier()  # 同步所有 GPU

    # 获取 FSDP 配置
    auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config)
    sharding_strategy = ShardingStrategy.FULL_SHARD  # ZeRO-3
    mixed_precision = MixedPrecision(...)  # bf16 计算, fp32 reduce

    fsdp_strategy = self.config.actor.strategy
    if fsdp_strategy == "fsdp":
        # ========== FSDP1（PyTorch < 2.4）==========
        # 文件: verl/workers/fsdp_workers.py:497-507
        actor_module_fsdp = FSDP(
            actor_module,  # 输入：完整模型（32GB）
            cpu_offload=None if role == "actor" else CPUOffload(offload_params=True),
            param_init_fn=init_fn,
            auto_wrap_policy=auto_wrap_policy,  # 按层包装
            device_id=get_device_id(),
            sharding_strategy=sharding_strategy,  # ZeRO-3
            mixed_precision=mixed_precision,     # bf16 计算
            sync_module_states=True,             # ← 关键：同步初始化状态
            device_mesh=self.device_mesh,
        )
        # ↓ FSDP 内部执行：
        # 1. 将完整模型（32GB）分片为 8 份（每份 4GB）
        # 2. All-Reduce 同步参数（确保所有 GPU 的初始参数一致）
        # 3. 释放本地完整参数，只保留分片（4GB）
        # 4. 转换为 bf16 → 分片从 4GB 降到 2GB
        #
        # 显存变化：
        # - 初始化前：32GB（fp32 完整模型）
        # - FSDP 中：32GB (原模型) + 2GB (分片) = 34GB（峰值）
        # - 初始化后：2GB（bf16 分片，原模型被释放）

    elif fsdp_strategy == "fsdp2":
        # ========== FSDP2（PyTorch >= 2.4）==========
        # 文件: verl/workers/fsdp_workers.py:509-532
        fsdp_kwargs = {
            "mesh": fsdp_mesh,
            "mp_policy": mp_policy,
            "offload_policy": cpu_offload,
            "reshard_after_forward": fsdp_config.reshard_after_forward,
        }
        # 先获取完整参数
        full_state = actor_module.state_dict()  # 32GB（fp32）
        # 应用 FSDP2（in-place 转换）
        apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
        # 加载参数到分片模型
        fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
        # 释放 full_state
        del full_state

        # 显存变化：
        # - 初始化前：32GB（fp32 完整模型）
        # - FSDP2 中：32GB (原模型) + 32GB (full_state) = 64GB（峰值）
        # - 初始化后：2GB（bf16 分片）

    log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)
    # 此时显存：~2GB (参数分片)

    # ========== 步骤 7: 创建优化器（仅 Actor）==========
    # 文件: verl/workers/fsdp_workers.py:542-589
    if role == "actor":
        actor_optimizer = optim.AdamW(
            actor_module_fsdp.parameters(),
            lr=optim_config.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        # 优化器状态（momentum + variance）：2GB + 2GB = 4GB

        actor_lr_scheduler = get_lr_scheduler(...)

    return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

# ========== 显存占用总结 ==========
# 初始化峰值（FSDP1）: 34GB（fp32 完整模型 + bf16 分片）
# 初始化峰值（FSDP2）: 64GB（fp32 完整模型 + fp32 full_state）
# 初始化后稳定: 2GB（bf16 分片） + 4GB（优化器） = 6GB
```

**关键发现**：

1. **是的，显存中短暂存在两份模型权重**：
   - FSDP 初始化前：完整模型（32GB fp32）
   - FSDP 初始化中：完整模型 + 分片模型（34-64GB 峰值）
   - FSDP 初始化后：仅分片模型（2GB bf16）

2. **FSDP 初始化位置**：`verl/workers/fsdp_workers.py:497-507` (FSDP1) 或 `530-532` (FSDP2)

3. **sync_module_states=True 的作用**：
   - 在 FSDP 初始化时，所有 GPU 的参数必须一致
   - `sync_module_states=True` 确保 rank 0 的参数广播到所有 GPU
   - 这是分布式训练的必要步骤

---

### 3.4 `_build_rollout()` - Rollout 模型加载

**代码位置**：`verl/workers/fsdp_workers.py:591-630`

#### 3.4.1 Rollout 构建入口

```python
# 文件: verl/workers/fsdp_workers.py:591-630

def _build_rollout(self):
    """构建 Rollout 模型（vLLM/SGLang）"""

    # ========== 步骤 1: 创建 Rollout Device Mesh ==========
    rollout_device_mesh = init_device_mesh(
        get_device_name(),
        mesh_shape=(self.world_size,),
        mesh_dim_names=("rollout",)
    )

    # ========== 步骤 2: 创建 Rollout Engine ==========
    rollout_config = self.config.rollout
    self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
        config=rollout_config,
        model_config=model_config,
        device_mesh=rollout_device_mesh
    )
    # → 内部调用 vLLM 或 SGLang 初始化
    # → 从磁盘加载完整模型（16GB bf16）
    # → 分配 KV Cache（32GB）

    log_gpu_memory_usage(f"After building {rollout_config.name} rollout", logger=logger)
    # 此时显存：16GB (参数) + 32GB (KV Cache) = 48GB

    return self.rollout
```

#### 3.4.2 `get_rollout_class()` - 动态选择 Rollout 引擎

**代码位置**：`verl/workers/rollout/base.py:88-93`

```python
# 文件: verl/workers/rollout/base.py:88-93

def get_rollout_class(rollout_name: str, mode: str = "sync") -> type[BaseRollout]:
    """根据配置动态返回 Rollout 引擎类"""
    from verl.workers.rollout.utils import ROLLOUT_MAPPING

    # ROLLOUT_MAPPING 定义：
    # {
    #     ("vllm", "sync"): vLLMRollout,
    #     ("vllm", "async"): vLLMRolloutAsync,
    #     ("sglang", "sync"): SGLangRollout,
    #     ("sglang", "async"): SGLangRollout,  # SGLang 统一处理 sync/async
    #     ("hf", "sync"): HFRollout,
    # }

    rollout_cls = ROLLOUT_MAPPING.get((rollout_name, mode), None)
    if rollout_cls is None:
        raise ValueError(f"Unsupported rollout: {rollout_name} with mode: {mode}")

    return rollout_cls

# 调用示例：
# get_rollout_class("vllm", "sync")  → 返回 vLLMRollout 类
# get_rollout_class("sglang", "sync")  → 返回 SGLangRollout 类
```

**关键点**：
- 返回的是**类（class）**，而不是实例（instance）
- 后续通过 `rollout_cls(config, model_config, device_mesh)` 实例化
- `mode="sync"` 表示同步模式，`mode="async"` 表示异步模式（用于 Agent Loop）

---

#### 3.4.3 vLLMRollout 初始化详解

**代码位置**：`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:92-237`

**核心流程**：

```python
# 文件: verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:92-237

class vLLMRollout(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        # ========== 步骤 1: 解析模型配置 ==========
        # 文件: verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:106-114
        actor_module = model_config.local_path  # 模型路径，如 "Qwen/Qwen3-8B"
        tokenizer = model_config.get_tokenizer()
        model_hf_config = model_config.hf_config  # HuggingFace config
        trust_remote_code = model_config.trust_remote_code
        lora_path = model_config.get("lora_path", None)  # LoRA 路径（可选）

        # ========== 步骤 2: 验证 max_model_len ==========
        # 文件: verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:116-161
        if not self.config.get("max_model_len", None):
            self.config.max_model_len = self.config.prompt_length + self.config.response_length

        # 检查模型的 max_position_embeddings 是否足够
        assert model_hf_config.max_position_embeddings >= self.config.max_model_len, (
            "model context length should be greater than total sequence length"
        )

        # ========== 步骤 3: 配置 vLLM Engine 参数 ==========
        # 文件: verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:163-186
        max_num_seqs = self.config.get("max_num_seqs", None)  # 最大并发序列数
        max_model_len = self.config.max_model_len
        load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format

        # 配置 torch.compile（可选，用于加速）
        compilation_config = None
        if hasattr(self.config, "vllm_compilation_config"):
            compilation_config = CompilationConfig.from_cli(
                backend=self.config.vllm_compilation_config.backend,
                custom_ops=self.config.vllm_compilation_config.custom_ops.split(","),
            )

        # ========== 步骤 4: 初始化 vLLM 推理引擎（核心！）==========
        # 文件: verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:188-210
        from vllm import LLM

        self.inference_engine = LLM(
            model=actor_module,  # 模型路径
            enable_sleep_mode=self.config.free_cache_engine,  # 是否启用 Sleep Mode（释放 KV Cache）
            tensor_parallel_size=tensor_parallel_size,  # TP 大小（例如 8）
            distributed_executor_backend="external_launcher",  # 使用外部启动器（Ray/torch.distributed）
            dtype=self.config.dtype,  # 数据类型（如 "bfloat16"）
            enforce_eager=self.config.enforce_eager,  # 是否强制 Eager 模式（禁用 CUDA Graph）
            gpu_memory_utilization=self.config.gpu_memory_utilization,  # GPU 显存利用率（如 0.6）
            max_model_len=max_model_len,  # 最大序列长度
            max_num_seqs=max_num_seqs,  # 最大并发序列数
            load_format=load_format,  # 加载格式（"auto", "dummy", "safetensors", etc.）
            enable_chunked_prefill=self.config.enable_chunked_prefill,  # 是否启用分块 Prefill
            enable_prefix_caching=self.config.enable_prefix_caching,  # 是否启用前缀缓存
            trust_remote_code=trust_remote_code,  # 是否信任远程代码
            compilation_config=compilation_config,  # torch.compile 配置
        )

        # vLLM 内部执行：
        # 1. 加载模型权重（完整 16GB bf16）
        # 2. 初始化 KV Cache Manager（分配 32GB GPU 显存）
        # 3. 创建 Scheduler（管理请求调度）
        # 4. 启动 Worker Processes（每个 GPU 一个 Worker）

        # 显存占用：
        # - 模型参数：16GB (bf16)
        # - KV Cache：32GB（根据 gpu_memory_utilization 自动分配）
        # - 总计：~48GB

        # ========== 步骤 5: 配置 SamplingParams（采样参数）==========
        # 文件: verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:212-227
        self.sampling_params = SamplingParams(
            n=1,  # 每个 prompt 生成 1 个序列
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.response_length,  # 最大生成长度
        )

        self.tokenizer = tokenizer
```

**vLLM 初始化的关键配置参数**：

| 参数 | 作用 | 典型值 |
|------|------|--------|
| `gpu_memory_utilization` | GPU 显存利用率 | 0.6（为训练预留 40% 显存） |
| `max_num_seqs` | 最大并发序列数 | 256（影响吞吐量） |
| `enable_chunked_prefill` | 分块 Prefill | True（减少延迟） |
| `enable_prefix_caching` | 前缀缓存 | True（加速 prompt 复用） |
| `enable_sleep_mode` | Sleep 模式 | True（允许动态释放 KV Cache） |
| `tensor_parallel_size` | Tensor 并行度 | 8（跨 8 个 GPU 分片） |

**显存占用时间线**：

```
初始化前（Trainer Mode）:
├── Actor FSDP 分片: 2GB
├── Actor 优化器: 4GB
└── 总计: 6GB

vLLM 初始化中:
├── 加载模型权重: +16GB (完整 bf16 模型)
├── 分配 KV Cache: +32GB (根据 gpu_memory_utilization)
└── 总计: 6GB + 48GB = 54GB

vLLM 初始化后（trainer_mode 释放 KV Cache）:
├── Actor FSDP 分片: 2GB
├── Actor 优化器: 4GB
├── vLLM 模型参数: 16GB (保留)
├── vLLM KV Cache: 0GB (已释放)
└── 总计: 22GB
```

---

#### 3.4.4 SGLangRollout 初始化详解

**代码位置**：`verl/workers/rollout/sglang_rollout/sglang_rollout.py:249-467`

**核心流程**：

```python
# 文件: verl/workers/rollout/sglang_rollout/sglang_rollout.py:249-467

class SGLangRollout(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        # ========== 步骤 1: 解析模型配置 ==========
        # 文件: verl/workers/rollout/sglang_rollout/sglang_rollout.py:258-273
        actor_module = model_config.local_path  # 模型路径
        processing_class = model_config.get_processor()  # Tokenizer 或 Processor（多模态）
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code

        # ========== 步骤 2: 初始化 Tools 和 Interactions（多轮对话/工具调用）==========
        # 文件: verl/workers/rollout/sglang_rollout/sglang_rollout.py:267-274
        (
            self._tool_schemas,      # 工具 Schema（OpenAI 格式）
            self._tool_map,          # 工具名称 → 工具实例映射
            self._tool_call_parser_type,  # 工具调用解析器类型
            self._sgl_tools,         # SGLang 格式的工具定义
            self._function_call_parser,  # 函数调用解析器
        ) = self._initialize_tools(config, processing_class)

        self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(config)

        # ========== 步骤 3: 初始化分布式环境 ==========
        # 文件: verl/workers/rollout/sglang_rollout/sglang_rollout.py:284
        self._init_distributed_env(device_mesh_cpu=None)

        # 内部执行：
        # 1. 解析 tensor_parallel_size（TP 大小）
        # 2. 创建 Device Mesh（CPU 和 GPU）
        # 3. 初始化 torch.distributed 进程组
        # 4. 设置环境变量（CUDA_VISIBLE_DEVICES, etc.）

        # ========== 步骤 4: 验证配置 ==========
        # 文件: verl/workers/rollout/sglang_rollout/sglang_rollout.py:286
        self._verify_config(model_hf_config=model_hf_config)

        # 检查 max_model_len 是否足够
        # 检查 max_assistant_turns 和 max_user_turns 配置

        # ========== 步骤 5: 初始化 SGLang 推理引擎（核心！）==========
        # 文件: verl/workers/rollout/sglang_rollout/sglang_rollout.py:288
        self._init_inference_engine(trust_remote_code, actor_module, port=None)

        # ↓ 详见下方 _init_inference_engine() 详解

        # ========== 步骤 6: 配置 SamplingParams ==========
        # 文件: verl/workers/rollout/sglang_rollout/sglang_rollout.py:290
        self._init_sampling_params()

        self.processing_class = processing_class
        self.pad_token_id = self.processing_class.pad_token_id


# ========== SGLang 引擎初始化详解 ==========
# 文件: verl/workers/rollout/sglang_rollout/sglang_rollout.py:392-467

def _init_inference_engine(self, trust_remote_code, actor_module, port):
    """初始化 SGLang 推理引擎"""

    # ========== 步骤 1: 计算节点和 TP 配置 ==========
    # 文件: verl/workers/rollout/sglang_rollout/sglang_rollout.py:394-407
    nnodes = -(-self._tp_size // len(self.visible_devices_set))  # 向上取整
    # 例如：TP=8, 每个节点 8 GPU → nnodes=1
    # 例如：TP=16, 每个节点 8 GPU → nnodes=2

    if nnodes > 1:
        # 多节点：需要配置分布式初始化地址
        ip = get_ip()
        port = get_open_port() if port is None else port
        dist_init_addr = f"{ip}:{port}"  # 例如：192.168.1.100:29500
    else:
        # 单节点：不需要分布式地址
        dist_init_addr = None

    # ========== 步骤 2: 配置 SGLang Engine 参数 ==========
    # 文件: verl/workers/rollout/sglang_rollout/sglang_rollout.py:409-463
    load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format
    tp_size_per_node = self._tp_size // nnodes  # 每个节点的 TP 大小
    node_rank = self._tp_rank // tp_size_per_node  # 当前进程所在节点的 rank
    first_rank_in_node = self._tp_rank % tp_size_per_node == 0  # 是否是节点内的第一个进程

    engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
    attention_backend = engine_kwargs.pop("attention_backend", None)  # 注意力后端（默认 fa3）
    max_running_requests = self.config.get("max_num_seqs", None)  # 最大并发请求数

    # ========== 步骤 3: 创建 SGLang AsyncEngine（只在 first_rank_in_node 创建）==========
    # 文件: verl/workers/rollout/sglang_rollout/sglang_rollout.py:429-475
    if first_rank_in_node:
        rank = dist.get_rank()
        backend = attention_backend if attention_backend is not None else "fa3"

        args = {
            "model_path": actor_module,  # 模型路径
            "dtype": self.config.dtype,  # 数据类型（如 "bfloat16"）
            "mem_fraction_static": self.config.gpu_memory_utilization,  # GPU 显存利用率
            "enable_memory_saver": True,  # 启用显存节省模式
            "base_gpu_id": 0,  # 基础 GPU ID
            "gpu_id_step": 1,  # GPU ID 步长
            "tp_size": self._tp_size,  # Tensor 并行大小
            "node_rank": node_rank,  # 节点 rank
            "load_format": load_format,  # 加载格式
            "dist_init_addr": dist_init_addr,  # 分布式初始化地址（多节点）
            "nnodes": nnodes,  # 节点数
            "trust_remote_code": trust_remote_code,  # 是否信任远程代码
            "max_running_requests": max_running_requests,  # 最大并发请求数
            "port": 30000 + rank,  # SGLang 内部通信端口
            "log_level": "info",  # 日志级别
            "mm_attention_backend": backend,  # 多模态注意力后端（fa3）
            "attention_backend": backend,  # 注意力后端（fa3）
            "skip_tokenizer_init": self.config.skip_tokenizer_init,  # 是否跳过 tokenizer 初始化
            "dist_timeout": 1800,  # 分布式超时时间（30分钟）
        }

        # 创建 SGLang AsyncEngine
        from sglang_rollout.async_engine import AsyncEngine
        self._engine = AsyncEngine(**args)
        # SGLang 内部执行：
        # 1. 加载模型权重（完整 16GB bf16）
        # 2. 初始化 KV Cache Manager（分配 32GB GPU 显存）
        # 3. 创建 Tokenizer Manager（管理 tokenization）
        # 4. 启动 Worker Processes（每个 GPU 一个 Worker）
        # 5. 初始化 Scheduler（管理请求调度和批处理）
    else:
        # 非 first_rank_in_node 的进程不创建 Engine，只等待同步
        self._engine = None

    # 显存占用：
    # - 模型参数：16GB (bf16)
    # - KV Cache：32GB（根据 mem_fraction_static 自动分配）
    # - 总计：~48GB
```

sglang_rollout中使用的后端Engine为自定义写的Rollout Engine：

```python
class AsyncEngine(sglang.srt.entrypoints.engine.Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def release_memory_occupation(self, tags: Optional[list[str]] = None):
        """Release GPU occupation temporarily."""
        if tags is None:
            obj = ReleaseMemoryOccupationReqInput()
        else:
            obj = ReleaseMemoryOccupationReqInput(tags=tags)
        return await self.tokenizer_manager.release_memory_occupation(obj, None)

    async def resume_memory_occupation(self, tags: Optional[list[str]] = None):
        """Resume GPU occupation."""
        if tags is None:
            obj = ResumeMemoryOccupationReqInput()
        else:
            obj = ResumeMemoryOccupationReqInput(tags=tags)
        return await self.tokenizer_manager.resume_memory_occupation(obj, None)

    async def update_weights_from_tensor(self, update_weights_request: UpdateWeightsFromTensorReqInput):
        return await self.tokenizer_manager.update_weights_from_tensor(update_weights_request, None)

    async def flush_cache(self):
        return await self.tokenizer_manager.flush_cache()

    async def abort_request(self, rid: str = "", abort_all: bool = False):
        """Abort a specific request or all requests.

        Args:
            rid: The request ID to abort. If empty and abort_all is False, no action is taken.
            abort_all: If True, abort all running requests regardless of rid.
        """
        return self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)

```

**SGLang 与 vLLM 的关键区别**：

| 特性 | vLLM | SGLang |
|------|------|--------|
| **多轮对话** | 有限支持 | 原生支持（Conversation Manager） |
| **工具调用** | 需要外部实现 | 内置 Function Call Parser |
| **注意力后端** | FlashAttention 2 | FlashAttention 3（更快） |
| **前缀缓存** | RadixAttention | RadixAttention |
| **异步模式** | LLMEngine.generate() | AsyncEngine（原生异步） |
| **模型加载** | LLM(...) | AsyncEngine(...) |
| **API 风格** | 同步为主 | 异步为主（async/await） |

**SGLang 特有功能（用于多轮对话）**：

```python
# 1. 工具调用解析器
self._function_call_parser = FunctionCallParser(
    sgl_tools,  # 工具列表
    tool_call_parser_type,  # 解析器类型（如 "hermes"）
)

# 2. 工具执行
tool_call_results = await asyncio.gather(*[
    self._tool_map[tool_call.function.name].execute(
        request_id,
        tool_call.function.arguments,
    )
    for tool_call in parsed_tool_calls
])

# 3. Interaction 机制（用于验证和反馈）
interaction = self.interaction_map[interaction_name]
should_terminate, content, reward, metrics = await interaction.generate_response(
    request_id, messages, **interaction_kwargs
)
```

**显存占用时间线**（与 vLLM 类似）：

```
初始化前（Trainer Mode）:
├── Actor FSDP 分片: 2GB
├── Actor 优化器: 4GB
└── 总计: 6GB

SGLang 初始化中:
├── 加载模型权重: +16GB (完整 bf16 模型)
├── 分配 KV Cache: +32GB (根据 mem_fraction_static)
└── 总计: 6GB + 48GB = 54GB

SGLang 初始化后（trainer_mode 释放 KV Cache）:
├── Actor FSDP 分片: 2GB
├── Actor 优化器: 4GB
├── SGLang 模型参数: 16GB (保留)
├── SGLang KV Cache: 0GB (已释放)
└── 总计: 22GB
```

---

#### 3.4.5 vLLM vs SGLang 总结

**共同点**：
- 都加载完整的 16GB 模型（不分片）
- 都分配 32GB KV Cache（可动态释放）
- 都与 Actor FSDP 共享同一 GPU
- 都支持 Sleep Mode（free_cache_engine=True）

**选择建议**：
- **单轮对话**：使用 vLLM（更成熟，社区支持好）
- **多轮对话 + 工具调用**：使用 SGLang（原生支持，更高效）
- **超大 Batch**：使用 vLLM（更好的批处理优化）
- **低延迟推理**：使用 SGLang（FA3 后端更快）

**配置示例**：

```bash
# 使用 vLLM
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
actor_rollout_ref.rollout.enable_prefix_caching=True

# 使用 SGLang（多轮对话）
actor_rollout_ref.rollout.name=sglang \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
actor_rollout_ref.rollout.multi_turn.enable=True \
actor_rollout_ref.rollout.multi_turn.tool_config_path=path/to/tool_config.yaml
```

---

#### 3.4.6 KV Cache 分配与释放机制

**核心问题**：Rollout 引擎在初始化时如何分配 KV Cache？训练时如何释放？

##### KV Cache 初始化分配

**vLLM KV Cache 分配**：

**代码位置**：vLLM 内部 `vllm/worker/worker.py` 和 `vllm/core/block_manager.py`

```python
# vLLM 初始化时的 KV Cache 分配流程：

# 1. 在 LLM.__init__() 中，vLLM 会初始化 Worker
# 文件: vllm/engine/llm_engine.py
class LLMEngine:
    def __init__(self, ...):
        # 创建 Worker（每个 GPU 一个）
        self.workers = self._create_workers()

        # 初始化 KV Cache（在每个 Worker 中）
        for worker in self.workers:
            worker.init_cache_engine(
                gpu_memory=gpu_memory_utilization,  # 例如 0.6（60% GPU 显存）
            )

# 2. Worker 初始化 KV Cache
# 文件: vllm/worker/worker.py
class Worker:
    def init_cache_engine(self, gpu_memory):
        # 计算可用显存
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = total_gpu_memory * gpu_memory

        # 减去模型参数占用（16GB）
        available_memory -= self.model_memory_usage  # 16GB

        # 计算 KV Cache 可以使用的显存
        kv_cache_memory = available_memory  # 约 32GB（80GB * 0.6 - 16GB）

        # 创建 Block Manager（管理 KV Cache 块）
        self.cache_engine = CacheEngine(
            cache_config=self.cache_config,
            model_config=self.model_config,
            parallel_config=self.parallel_config,
        )

        # 分配 KV Cache 块（PagedAttention）
        # 每个块大小：block_size（如 16） * num_layers（如 32） * num_heads（如 32） * head_dim（如 128）
        # 单个块大小：16 * 32 * 32 * 128 * 2 (K+V) * 2 bytes (bf16) = 512KB
        # 总块数：32GB / 512KB = 64K 个块
        self.cache_engine.allocate_blocks(kv_cache_memory)

# 3. KV Cache 的物理结构（GPU 显存布局）
┌─────────────────────────────────────────────────────────┐
│ GPU 显存（80GB）                                        │
├─────────────────────────────────────────────────────────┤
│ 1. 模型参数: 16GB (bf16)                               │
├─────────────────────────────────────────────────────────┤
│ 2. KV Cache 块池: 32GB                                 │
│    ├── Block 0: [K0, V0] (512KB)                       │
│    ├── Block 1: [K1, V1] (512KB)                       │
│    ├── Block 2: [K2, V2] (512KB)                       │
│    ├── ...                                              │
│    └── Block 64K: [K_64K, V_64K] (512KB)               │
├─────────────────────────────────────────────────────────┤
│ 3. 激活值缓冲: 动态分配                                │
├─────────────────────────────────────────────────────────┤
│ 4. 其他（Scheduler, Metadata, etc.): ~2GB              │
└─────────────────────────────────────────────────────────┘

# 4. PagedAttention 的 KV Cache 管理
# - 将 KV Cache 分为固定大小的块（blocks）
# - 每个序列动态分配所需的块
# - 序列完成后，块被释放回池中
# - 优点：减少显存碎片，提高利用率
```

**SGLang KV Cache 分配**：

**代码位置**：SGLang 内部 `sglang/srt/managers/scheduler.py`

```python
# SGLang 初始化时的 KV Cache 分配流程：

# 1. 在 AsyncEngine.__init__() 中，SGLang 会初始化 Scheduler
# 文件: sglang/srt/entrypoints/engine.py
class Engine:
    def __init__(self, ...):
        # 创建 Tokenizer Manager
        self.tokenizer_manager = TokenizerManager(...)

        # Tokenizer Manager 内部会初始化 Scheduler
        # Scheduler 负责管理 KV Cache

# 2. Scheduler 初始化 KV Cache
# 文件: sglang/srt/managers/scheduler.py
class Scheduler:
    def __init__(self, server_args, ...):
        # 计算可用显存（与 vLLM 类似）
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = total_memory * server_args.mem_fraction_static  # 0.6

        # 减去模型参数占用
        available_memory -= self.model_memory_usage  # 16GB

        # 创建 Memory Pool（RadixAttention 的 KV Cache 管理）
        self.memory_pool = ReqToTokenPool(
            size=available_memory,  # 32GB
            max_context_len=server_args.max_model_len,
            device=self.device,
        )

        # RadixAttention: 基于前缀树（Trie）的 KV Cache 管理
        # - 共享相同前缀的 KV Cache（减少重复计算）
        # - 动态分配和回收 Token 级别的 KV Cache
        # - 比 PagedAttention 更细粒度的管理

# 3. SGLang KV Cache 的物理结构
┌─────────────────────────────────────────────────────────┐
│ GPU 显存（80GB）                                        │
├─────────────────────────────────────────────────────────┤
│ 1. 模型参数: 16GB (bf16)                               │
├─────────────────────────────────────────────────────────┤
│ 2. KV Cache Token Pool: 32GB                           │
│    ├── Token 0: [K0, V0] (256 bytes, per layer)        │
│    ├── Token 1: [K1, V1] (256 bytes)                   │
│    ├── Token 2: [K2, V2] (256 bytes)                   │
│    ├── ...                                              │
│    └── Token N: [KN, VN] (256 bytes)                   │
│    (N = 32GB / (256 bytes * 32 layers) ≈ 4M tokens)    │
├─────────────────────────────────────────────────────────┤
│ 3. Radix Tree (前缀树): 管理 Token 共享               │
├─────────────────────────────────────────────────────────┤
│ 4. 其他（Scheduler, Metadata, etc.): ~2GB              │
└─────────────────────────────────────────────────────────┘
```

---

##### KV Cache 训练时释放

**关键时机**：在 `trainer_mode()` 中释放 KV Cache，为训练腾出显存。

**代码位置**：`verl/workers/fsdp_workers.py:741-757`

```python
# 文件: verl/workers/fsdp_workers.py:741-757

async def trainer_mode(self):
    """Context switch hybridengine to trainer mode."""

    # ========== 释放 KV Cache（关键！）==========
    if self.config.rollout.free_cache_engine:
        log_gpu_memory_usage("Before rollout offload", logger=logger)
        await self.rollout.release()  # ← 调用 Rollout 引擎的 release()
        log_gpu_memory_usage("After rollout offload", logger=logger)

    # 其他步骤...
```

**vLLM KV Cache 释放实现**：

**代码位置**：`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` 和 vLLM 内部

```python
# 文件: verl/workers/rollout/base.py (BaseRollout 基类)

async def release(self):
    """Release KV Cache and other caches."""
    if self.device_mesh["infer_tp"].get_local_rank() == 0:
        # 只在 TP rank 0 执行（避免重复）
        await self._release_kv_cache()

async def _release_kv_cache(self):
    """vLLM 的 KV Cache 释放"""
    # 如果 vLLM 支持 Sleep Mode (enable_sleep_mode=True)
    if hasattr(self.inference_engine, 'sleep'):
        # 调用 vLLM 的 sleep() 方法
        await self.inference_engine.sleep()
        # ↓ vLLM 内部执行：
        # 1. 释放所有 KV Cache 块（32GB）
        # 2. 保留模型参数（16GB）
        # 3. 清理 Scheduler 状态
    else:
        # 旧版 vLLM 或不支持 Sleep Mode
        # 只清空 KV Cache，但不释放显存
        self.inference_engine.cache_engine.clear()

# vLLM Sleep Mode 内部实现（简化版）：
# 文件: vllm/engine/llm_engine.py
class LLMEngine:
    async def sleep(self):
        # 1. 清空所有正在运行的请求
        self.scheduler.abort_all_requests()

        # 2. 释放 KV Cache 块
        self.cache_engine.free_all_blocks()
        # → 将 32GB KV Cache 显存标记为可用
        # → torch.cuda.empty_cache() 整理显存碎片

        # 3. 清理 Scheduler 状态
        self.scheduler.reset()

        # 4. 保留模型参数（不释放）
        # self.model.parameters() 仍然在 GPU (16GB)

# 释放后的显存状态：
# - 模型参数: 16GB (保留)
# - KV Cache: 0GB (已释放)
# - 总计: 16GB
```

**SGLang KV Cache 释放实现**：

**代码位置**：`verl/workers/rollout/sglang_rollout/sglang_rollout.py:1500-1503`

```python
# 文件: verl/workers/rollout/sglang_rollout/sglang_rollout.py:1500-1503

async def release(self):
    """Release weights and kv cache in GPU memory."""
    if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
        await self._engine.release_memory_occupation(tags=["kv_cache", "weights"])
        # ↓ 调用 SGLang Engine 的释放接口

# SGLang Engine 释放接口：
# 文件: sglang/srt/entrypoints/engine.py (AsyncEngine)
class AsyncEngine:
    async def release_memory_occupation(self, tags: list[str]):
        """释放指定的显存占用"""
        # 构造释放请求
        obj = ReleaseMemoryOccupationReqInput(tags=tags)

        # 发送给 Tokenizer Manager 处理
        return await self.tokenizer_manager.release_memory_occupation(obj, None)

# Tokenizer Manager 处理释放请求：
# 文件: sglang/srt/managers/tokenizer_manager.py
class TokenizerManager:
    async def release_memory_occupation(self, req: ReleaseMemoryOccupationReqInput):
        if "kv_cache" in req.tags:
            # 1. 清空 Scheduler 的 Memory Pool
            self.scheduler.memory_pool.clear()
            # → 释放所有 Token 的 KV Cache（32GB）
            # → 清空 RadixTree（前缀树）

            # 2. 清理所有正在运行的请求
            self.scheduler.abort_all_requests()

        if "weights" in req.tags:
            # 释放模型权重（通常不释放，只释放 KV Cache）
            pass

        # 3. 整理显存碎片
        torch.cuda.empty_cache()

# 释放后的显存状态：
# - 模型参数: 16GB (保留)
# - KV Cache: 0GB (已释放)
# - Radix Tree: 0GB (已清空)
# - 总计: 16GB
```

---

##### KV Cache 重新分配（Rollout Mode）

**代码位置**：`verl/workers/fsdp_workers.py:658-739`

```python
# 文件: verl/workers/fsdp_workers.py:658-739

async def rollout_mode(self):
    """Context switch hybridengine to rollout mode."""

    # ... (前面的参数同步步骤)

    # ========== 恢复 KV Cache ==========
    if self.config.rollout.free_cache_engine:
        await self.rollout.resume(tags=["kv_cache"])
        # ↓ 重新分配 KV Cache（32GB）

    log_gpu_memory_usage("After resume kv_cache", logger=logger)
```

**vLLM KV Cache 恢复**：

```python
# vLLM resume() 实现（简化版）：
class LLMEngine:
    async def resume(self):
        # 1. 重新分配 KV Cache 块
        self.cache_engine.allocate_blocks(self.kv_cache_memory)
        # → 重新分配 32GB KV Cache 显存

        # 2. 初始化 Scheduler
        self.scheduler.reset()

        # 3. 准备接受新请求
        self.is_ready = True

# 恢复后的显存状态：
# - 模型参数: 16GB (已同步最新参数)
# - KV Cache: 32GB (重新分配)
# - 总计: 48GB
```

**SGLang KV Cache 恢复**：

**代码位置**：`verl/workers/rollout/sglang_rollout/sglang_rollout.py:1491-1498`

```python
# 文件: verl/workers/rollout/sglang_rollout/sglang_rollout.py:1491-1498

async def resume(self, tags: list[str]):
    """Resume rollout weights or kv cache in GPU memory."""
    if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
        await self._engine.resume_memory_occupation(tags=tags)

# SGLang resume() 实现：
class AsyncEngine:
    async def resume_memory_occupation(self, tags: list[str]):
        obj = ResumeMemoryOccupationReqInput(tags=tags)
        return await self.tokenizer_manager.resume_memory_occupation(obj, None)

# Tokenizer Manager 处理恢复请求：
class TokenizerManager:
    async def resume_memory_occupation(self, req: ResumeMemoryOccupationReqInput):
        if "kv_cache" in req.tags:
            # 1. 重新初始化 Memory Pool
            self.scheduler.memory_pool.reset(
                size=self.kv_cache_memory,  # 32GB
                max_context_len=self.max_model_len,
            )
            # → 重新分配 Token Pool（32GB）

            # 2. 重建 RadixTree
            self.scheduler.tree_cache = RadixCache(...)

        if "weights" in req.tags:
            # 恢复模型权重（通常已在显存中）
            pass

# 恢复后的显存状态：
# - 模型参数: 16GB (已同步最新参数)
# - KV Cache: 32GB (重新分配)
# - Radix Tree: 已重建
# - 总计: 48GB
```

---

##### KV Cache 分配/释放完整时间线

```
┌────────────────────────────────────────────────────────────┐
│ PPO 单次迭代的 KV Cache 显存变化（GPU 0）                 │
└────────────────────────────────────────────────────────────┘

时刻 0s: Trainer Mode (初始)
├── Actor FSDP 分片: 2GB
├── Actor 优化器: 4GB
├── Rollout 参数: 16GB
├── Rollout KV Cache: 0GB (已释放)
└── 总计: 22GB

时刻 0-1s: generate_sequences() → rollout_mode()
├── 参数同步: Actor FSDP → Rollout (临时 +16GB)
├── resume KV Cache: +32GB
│   ├── vLLM: allocate_blocks(32GB)
│   └── SGLang: memory_pool.reset(32GB)
└── 总计: 22GB + 32GB = 54GB

时刻 1-2s: Rollout 推理（generate_sequences）
├── KV Cache 动态使用:
│   ├── 生成第 1 个 token: 使用 1 block (512KB)
│   ├── 生成第 2 个 token: 使用 2 blocks (1MB)
│   ├── ...
│   └── 生成第 N 个 token: 使用 N blocks
└── 总计: ~50GB (部分 KV Cache 被占用)

时刻 2s: trainer_mode()
├── release KV Cache: -32GB
│   ├── vLLM: sleep() → free_all_blocks()
│   └── SGLang: memory_pool.clear()
└── 总计: 50GB - 32GB = 18GB

时刻 2-3s: compute_log_prob() (Trainer Mode)
├── 使用 Actor FSDP 计算 log prob
├── 不需要 KV Cache (Teacher Forcing)
└── 总计: 18-22GB

时刻 3-6s: update_actor() (Trainer Mode)
├── FSDP Forward + Backward
├── 不需要 KV Cache
└── 总计: 22-38GB (取决于 batch size)

时刻 6s: 迭代结束，准备下一次 rollout
└── 总计: 22GB (回到初始状态)
```

**关键优化配置**：

```bash
# 启用 KV Cache 动态释放（必须）
actor_rollout_ref.rollout.free_cache_engine=True

# 调整 GPU 显存利用率（控制 KV Cache 大小）
actor_rollout_ref.rollout.gpu_memory_utilization=0.6  # 60%
# → KV Cache ≈ (80GB * 0.6 - 16GB) = 32GB

# 减少 KV Cache 以为训练预留更多显存
actor_rollout_ref.rollout.gpu_memory_utilization=0.4  # 40%
# → KV Cache ≈ (80GB * 0.4 - 16GB) = 16GB

# vLLM 特有：启用前缀缓存（减少重复计算）
actor_rollout_ref.rollout.enable_prefix_caching=True

# vLLM 特有：启用分块 Prefill（减少峰值显存）
actor_rollout_ref.rollout.enable_chunked_prefill=True
```

**总结**：
- **初始化时**：根据 `gpu_memory_utilization` 自动计算并分配 KV Cache（如 32GB）
- **训练时**：调用 `release()` 释放 KV Cache，保留模型参数（释放 32GB，保留 16GB）
- **推理时**：调用 `resume()` 重新分配 KV Cache（重新分配 32GB）
- **核心机制**：vLLM 使用 PagedAttention（块级管理），SGLang 使用 RadixAttention（Token 级管理 + 前缀树共享）

---

### 3.5 初始化后的模式状态

**代码位置**：`verl/workers/fsdp_workers.py:650-656`

```python
# 文件: verl/workers/fsdp_workers.py:650-656

# 5. switch to trainer mode
# NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
if rollout_config.mode == "sync" and self._is_actor:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(self.trainer_mode())
    # ↓ 切换到 Trainer 模式
```

**初始化完整流程**：

```
1. __init__() → 初始化分布式环境（~100MB）
   ↓
2. init_model() → 调用 _build_model_optimizer()
   ↓
3. _build_model_optimizer():
   ├── 加载完整 Actor 模型（32GB fp32）
   ├── FSDP 初始化（峰值 34-64GB）
   ├── 释放完整模型，保留分片（2GB bf16）
   └── 创建优化器（4GB）
   ↓ 此时显存：6GB

4. _build_rollout():
   ├── 加载 Rollout 模型（16GB bf16）
   └── 分配 KV Cache（32GB）
   ↓ 此时显存：6GB + 48GB = 54GB

5. trainer_mode():
   ├── 释放 KV Cache（-32GB）
   └── 保留 Rollout 参数（16GB）
   ↓ 最终显存：6GB + 16GB = 22GB
```

**初始化后的显存占用（每个 GPU）**：

```
GPU 0 (Trainer Mode):
├── Actor FSDP 参数分片: 2GB (bf16)
├── Actor 优化器状态: 4GB (fp32)
├── Rollout 参数（完整）: 16GB (bf16)
├── Rollout KV Cache: 0GB (已释放)
└── 总计: ~22GB / 80GB
```

### 3.2 初始化后的模式状态

**关键点**：初始化后，系统处于 **Trainer Mode**，而不是 Rollout Mode！

**代码位置**：`verl/workers/fsdp_workers.py:650-656`

```python
# 文件: verl/workers/fsdp_workers.py:650-656

# 5. switch to trainer mode
# NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
# For sync mode, we directly switch to trainer mode here.
# For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.
if rollout_config.mode == "sync" and self._is_actor:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(self.trainer_mode())  # ← 这里切换到 Trainer Mode
```

**初始化流程**：

```
1. _build_model_optimizer() → Actor FSDP 加载（2GB 参数分片 + 4GB 优化器）
2. _build_rollout() → Rollout 模型加载（16GB 完整参数 + 32GB KV Cache）
   ↓ 此时显存: ~54GB
3. trainer_mode() 被调用:
   ↓ rollout.release() 释放 KV Cache（-32GB）
   ↓ 此时显存: ~22GB
4. 初始化完成，系统处于 Trainer Mode
```

**为什么初始化后要切换到 Trainer Mode？**

根据代码注释：
> It's critical that hybrid engine in trainer mode initially to load checkpoint.

原因：
1. **加载 checkpoint** 时需要 Actor FSDP 在 GPU（用于加载参数）
2. Rollout KV Cache 在初始化时会占用大量显存，需要释放
3. 训练开始前可能需要执行验证（validation），需要 Trainer Mode

---

## 4. PPO 完整训练迭代流程

### 4.1 训练循环入口

**代码位置**：`verl/trainer/ppo/ray_trainer.py:962-1259`

```python
# 文件: verl/trainer/ppo/ray_trainer.py:962-1259

def fit(self):
    """
    The training loop of PPO.
    The driver process only need to call the compute functions of the worker group through RPC
    to construct the PPO dataflow.
    The light-weight advantage computation is done on the driver process.
    """

    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            # ========== 阶段 1: 生成序列（Rollout Phase）==========
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

            # ========== 阶段 2: 计算 Reward（TaskRunner CPU）==========
            reward_tensor, reward_extra_infos = compute_reward(batch, self.reward_fn)

            # ========== 阶段 3: 计算 Old Log Prob（Actor FSDP）==========
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

            # ========== 阶段 4: 计算 Advantage（TaskRunner CPU）==========
            batch = compute_advantage(batch, adv_estimator="grpo", ...)

            # ========== 阶段 5: 更新 Actor 模型（FSDP 训练）==========
            actor_output = self.actor_rollout_wg.update_actor(batch)

            # 下一轮迭代...
```

### 4.2 阶段 1：生成序列（Rollout Phase）

**代码位置**：
- 调用入口：`verl/trainer/ppo/ray_trainer.py:1042-1050`
- Worker 实现：`verl/workers/fsdp_workers.py:927-984`

```python
# ========== TaskRunner 调用 ==========
# 文件: verl/trainer/ppo/ray_trainer.py:1042-1050

gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)


# ========== 在每个 GPU Worker 中执行 ==========
# 文件: verl/workers/fsdp_workers.py:927-984

@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
@DistProfiler.annotate(color="red", role="rollout_generate")
def generate_sequences(self, prompts: DataProto):
    """生成序列（Rollout 阶段）"""

    # ========== 步骤 1.1: 切换到 Rollout 模式 ==========
    # 文件: verl/workers/fsdp_workers.py:945-950
    if self._is_actor:  # For rollout only, we do not switch context.
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.rollout_mode())  # ← 切换到 Rollout Mode
        # → Actor 权重同步到 Rollout Engine
        # → Actor 模型 offload 到 CPU（如果配置了）
        log_gpu_memory_usage("After switch to rollout mode", logger=logger)

    # ========== 步骤 1.2: 使用 Rollout 推理 ==========
    # 文件: verl/workers/fsdp_workers.py:952-957
    with simple_timer("generate_sequences", timing_generate):
        output = self.rollout.generate_sequences(prompts=prompts)
        # → vLLM/SGLang 执行推理
        # → 返回：sequences, token_level_scores, ...

    # ========== 步骤 1.3: 切换回 Trainer 模式 ==========
    # 文件: verl/workers/fsdp_workers.py:960-964
    if self._is_actor:
        loop.run_until_complete(self.trainer_mode())  # ← 切换回 Trainer Mode
        # → Rollout Engine offload
        # → Actor 模型加载回 GPU
        log_gpu_memory_usage("After switch to trainer mode", logger=logger)

    return output
```

### 4.3 阶段 2：计算 Reward（TaskRunner CPU）

**代码位置**：`verl/trainer/ppo/ray_trainer.py:1088-1098`

```python
# 文件: verl/trainer/ppo/ray_trainer.py:1088-1098

# ========== 在 TaskRunner（CPU）中计算 Reward ==========
# 对于 GRPO，使用规则 Reward（如 GSM8K 答案匹配）

with marked_timer("reward", timing_raw, color="yellow"):
    # compute reward model score
    if self.use_rm and "rm_scores" not in batch.batch.keys():
        reward_tensor = self.rm_wg.compute_rm_score(batch)
        batch = batch.union(reward_tensor)

    if self.config.reward_model.launch_reward_fn_async:
        future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
    else:
        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

# 不需要模式切换
# 不占用 GPU 显存
```

### 4.4 阶段 3：计算 Old Log Prob（使用 Actor FSDP）

**代码位置**：
- 调用入口：`verl/trainer/ppo/ray_trainer.py:1100-1110`
- Worker 实现：`verl/workers/fsdp_workers.py:986-1026`

```python
# 文件: verl/trainer/ppo/ray_trainer.py:1100-1110

# recompute old_log_probs
with marked_timer("old_log_prob", timing_raw, color="blue"):
    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
    entropys = old_log_prob.batch["entropys"]
    response_masks = batch.batch["response_mask"]
    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
    old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
    metrics.update(old_log_prob_metrics)
    old_log_prob.batch.pop("entropys")
    batch = batch.union(old_log_prob)


# 文件: verl/workers/fsdp_workers.py:986-1026

@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
@DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
def compute_log_prob(self, data: DataProto):
    """计算旧策略的 Log Probs（用于 PPO importance sampling）"""

    # ========== 不需要模式切换！==========
    # 因为当前已经在 Trainer Mode（上一步 trainer_mode() 完成）

    # ========== 步骤 3.1: 加载 Actor FSDP 到 GPU（如果 offload 了）==========
    if self._is_offload_param:
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
        # → 从 CPU 加载参数分片（2GB）

    # ========== 步骤 3.2: 使用 Actor FSDP 计算 log prob ==========
    output, entropys = self.actor.compute_log_prob(
        data=data,
        calculate_entropy=True
    )
    # → Forward pass（Teacher Forcing）
    # → 计算每个 token 的 log probability
    # → 使用的是 **当前训练中的 Actor 模型**（不是 Rollout）

    # ========== 步骤 3.3: Offload Actor FSDP 回 CPU（如果配置了）==========
    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        # → 释放 2GB 显存

    return output
```

### 4.5 阶段 4：计算 Advantage（TaskRunner CPU）

**代码位置**：`verl/trainer/ppo/ray_trainer.py:1132-1164`

```python
# 文件: verl/trainer/ppo/ray_trainer.py:1132-1164

with marked_timer("adv", timing_raw, color="brown"):
    # we combine with rule-based rm
    reward_extra_infos_dict: dict[str, list]
    if self.config.reward_model.launch_reward_fn_async:
        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
    batch.batch["token_level_scores"] = reward_tensor

    if reward_extra_infos_dict:
        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

    # compute rewards. apply_kl_penalty if available
    if self.config.algorithm.use_kl_in_reward:
        batch, kl_metrics = apply_kl_penalty(
            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
        )
        metrics.update(kl_metrics)
    else:
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

    # compute advantages, executed on the driver process
    norm_adv_by_std_in_grpo = self.config.algorithm.get(
        "norm_adv_by_std_in_grpo", True
    )  # GRPO adv normalization factor

    batch = compute_advantage(
        batch,
        adv_estimator=self.config.algorithm.adv_estimator,
        gamma=self.config.algorithm.gamma,
        lam=self.config.algorithm.lam,
        num_repeat=self.config.actor_rollout_ref.rollout.n,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=self.config.algorithm,
    )

# 在 TaskRunner（CPU）中执行
# 不需要模式切换
# 不占用 GPU 显存
```

### 4.6 阶段 5：更新 Actor 模型（FSDP 训练）

**代码位置**：
- 调用入口：`verl/trainer/ppo/ray_trainer.py:1174-1180`
- Worker 实现：`verl/workers/fsdp_workers.py:877-923`
- Actor 更新实现：`verl/workers/actor/dp_actor.py:359-503`

```python
# 文件: verl/trainer/ppo/ray_trainer.py:1174-1180

# implement critic warmup
if self.config.trainer.critic_warmup <= self.global_steps:
    # update actor
    with marked_timer("update_actor", timing_raw, color="red"):
        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
        actor_output = self.actor_rollout_wg.update_actor(batch)
    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
    metrics.update(actor_output_metrics)


# 文件: verl/workers/fsdp_workers.py:877-923

@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
@DistProfiler.annotate(color="red", role="actor_update")
def update_actor(self, data: DataProto):
    """执行 PPO 策略更新"""

    # ========== 不需要模式切换！==========
    # 因为当前已经在 Trainer Mode

    # ========== 步骤 5.1: 加载模型和优化器到 GPU ==========
    if self._is_offload_param:
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
        # → 加载 Actor 参数分片（2GB）
    if self._is_offload_optimizer:
        load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())
        # → 加载优化器状态（4GB）

    # ========== 步骤 5.2: 调用 PPO Actor 训练 ==========
    with self.ulysses_sharding_manager:
        data = data.to("cpu")  # 先移到 CPU，后续按 micro-batch 移到 GPU
        output = self.actor.update_policy(data)
        # ↓ 内部执行 mini-batch 训练循环
        # ↓ Forward, Backward, Optimizer Step
        # ↓ FSDP 自动同步梯度和参数（NCCL All-Reduce）
        # ↓ 详细流程见下方 update_policy() 详解

    # ========== 步骤 5.3: Offload 模型和优化器回 CPU ==========
    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        # → 释放 2GB
    if self._is_offload_optimizer:
        offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        # → 释放 4GB

    return output
```

---

#### 4.6.1 `update_policy()` - GRPO 更新详解（重点）

**代码位置**：`verl/workers/actor/dp_actor.py:359-503`

这是 **GRPO 训练的核心**，展示了哪些 tensor 参与反向传播，以及 FSDP 如何计算梯度。

##### 输入数据（从 DataProto 提取）

```python
# 文件: verl/workers/actor/dp_actor.py:365-384

select_keys = [
    "responses",          # 生成的响应 tokens (bs, response_length)
    "response_mask",      # 响应部分的 mask (bs, response_length)
    "input_ids",          # 完整输入 = prompt + response (bs, seq_length)
    "attention_mask",     # 注意力 mask (bs, seq_length)
    "position_ids",       # 位置编码 (bs, seq_length)
    "old_log_probs",      # ← 旧策略的 log probs (bs, response_length) [不需要梯度]
    "advantages",         # ← 优势估计 (bs, response_length) [不需要梯度]
]
if self.config.use_kl_loss:
    select_keys.append("ref_log_prob")  # ← 参考策略的 log probs [不需要梯度]
```

**关键区分**：
- **old_log_probs**：来自 `compute_log_prob()` 阶段（使用旧策略 forward 得到）
- **advantages**：来自 `compute_advantage()` 阶段（CPU 上计算，使用 GRPO 公式）
- 这两者都 **不参与梯度计算**（已经 detach）

---

##### Mini-Batch 和 Micro-Batch 切分

```python
# 文件: verl/workers/actor/dp_actor.py:388-402

# 切分为 mini-batch（PPO 标准做法）
mini_batches = data.split(self.config.ppo_mini_batch_size)  # 例如：512 → 4 个 128

for _ in range(self.config.ppo_epochs):  # 例如：2 个 epoch
    for batch_idx, mini_batch in enumerate(mini_batches):
        # 进一步切分为 micro-batch（用于梯度累积）
        if self.config.use_dynamic_bsz:
            # 动态批大小：根据 token 数量切分
            micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
        else:
            # 固定批大小：例如 128 → 8 个 16
            micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

        self.actor_optimizer.zero_grad()  # 清空梯度
```

**为什么需要两层切分？**
1. **Mini-Batch**：PPO 算法要求（见论文 https://arxiv.org/abs/1707.06347）
2. **Micro-Batch**：梯度累积（避免单个 batch 过大导致 OOM）

---

##### Forward Pass（FSDP 计算）

```python
# 文件: verl/workers/actor/dp_actor.py:406-428

for micro_batch in micro_batches:
    micro_batch = micro_batch.to(get_device_id())  # 移到 GPU
    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
    response_mask = model_inputs["response_mask"]
    old_log_prob = model_inputs["old_log_probs"]    # ← 不需要梯度
    advantages = model_inputs["advantages"]          # ← 不需要梯度

    # ========== 关键：Forward Pass ==========
    # 计算当前策略的 log probabilities（需要梯度）
    entropy, log_prob = self._forward_micro_batch(
        model_inputs,
        temperature=temperature,
        calculate_entropy=calculate_entropy
    )
    # → 返回：
    #   - log_prob: (bs, response_length) ← **需要梯度！**
    #   - entropy: (bs, response_length) ← 可选，用于 entropy bonus
```

**`_forward_micro_batch()` 内部流程**（`verl/workers/actor/dp_actor.py:85-272`）：

```python
# 简化伪代码

with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
    # 1. 准备输入
    input_ids = micro_batch["input_ids"]  # (bs, seq_length)
    attention_mask = micro_batch["attention_mask"]
    position_ids = micro_batch["position_ids"]

    # 2. FSDP 模型 Forward（关键！）
    output = self.actor_module(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,  # 训练模式，不使用 KV Cache
    )
    # → FSDP 内部流程：
    #   1. All-Gather 聚合完整参数（临时，16GB）
    #   2. 执行 forward（计算 logits）
    #   3. 释放临时聚合的参数（保留分片，2GB）
    #
    # → **详细计算过程**（从 input_ids 到 logits 的每一步）：
    #   请查看完整文档：FSDP_Forward_详解.md
    #   包括：
    #   - Embedding Layer（词嵌入）的具体计算和数值示例
    #   - 28 层 Transformer 的逐层详细计算
    #   - Self-Attention 的每个步骤（Q/K/V 计算、RoPE、Attention Scores、Softmax）
    #   - FFN（Feed-Forward Network）的 SwiGLU 激活函数
    #   - LM Head（输出层）的 logits 计算
    #   - 每一步的 tensor 维度变化
    #   - 每一步的显存占用和时间线

    # 3. 计算 log probabilities
    logits = output.logits  # (bs, seq_length, vocab_size)
    logits = logits[:, -response_length-1:-1, :]  # 取响应部分（去掉最后一个 token）
    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
    # → log_probs: (bs, response_length) ← **需要梯度！**

    # 4. 可选：计算 entropy
    if calculate_entropy:
        entropy = entropy_from_logits(logits)  # (bs, response_length)

    return entropy, log_probs
```

**FSDP Forward 的显存变化**：
```
Forward 开始前:
├── Actor FSDP 分片: 2GB (本地分片)
└── 总计: 2GB

Forward 中（FSDP All-Gather）:
├── Actor FSDP 分片: 2GB (保留)
├── All-Gather 临时参数: +16GB (完整参数)
├── Forward 激活值: +4GB (取决于 batch size)
└── 总计: 22GB ← 临时峰值

Forward 后:
├── Actor FSDP 分片: 2GB (保留)
├── Forward 激活值: 4GB (保留，用于 backward)
└── 总计: 6GB
```

---

##### Loss 计算（Policy Loss）

```python
# 文件: verl/workers/actor/dp_actor.py:430-478

# ========== 计算 Policy Loss ==========

if on_policy:  # 如果是 on-policy（GRPO 通常不是）
    old_log_prob = log_prob.detach()
else:
    old_log_prob = model_inputs["old_log_probs"]  # ← 使用预计算的 old_log_probs

# 选择 loss 函数（vanilla PPO、GPG、GSPO、etc.）
loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
policy_loss_fn = get_policy_loss_fn(loss_mode)
# → 对于 GRPO，通常使用 "vanilla" 或 "gpg"

# 提取可选的 importance sampling weights（如果使用）
rollout_is_weights = model_inputs.get("rollout_is_weights", None)

# ========== 核心：计算 Policy Loss ==========
pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
    old_log_prob=old_log_prob,    # ← 不需要梯度
    log_prob=log_prob,             # ← **需要梯度！**
    advantages=advantages,         # ← 不需要梯度
    response_mask=response_mask,   # ← mask
    loss_agg_mode=loss_agg_mode,
    config=self.config,
    rollout_is_weights=rollout_is_weights,
)
# → pg_loss: 标量 Tensor ← **需要梯度！**

# ========== 可选：Entropy Loss ==========
if entropy_coeff != 0:
    entropy_loss = agg_loss(
        loss_mat=entropy,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode
    )
    policy_loss = pg_loss - entropy_loss * entropy_coeff
else:
    policy_loss = pg_loss

# ========== 可选：KL Loss（GRPO 中通常使用）==========
if self.config.use_kl_loss:
    ref_log_prob = model_inputs["ref_log_prob"]  # ← 不需要梯度
    kld = kl_penalty(
        logprob=log_prob,          # ← **需要梯度！**
        ref_logprob=ref_log_prob,  # ← 不需要梯度
        kl_penalty=self.config.kl_loss_type
    )
    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
    # → policy_loss: 标量 Tensor ← **最终需要梯度！**
```

**Policy Loss 函数示例**（Vanilla PPO，`verl/trainer/ppo/core_algos.py:885-968`）：

```python
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,  # (bs, response_length) [不需要梯度]
    log_prob: torch.Tensor,       # (bs, response_length) [**需要梯度**]
    advantages: torch.Tensor,     # (bs, response_length) [不需要梯度]
    response_mask: torch.Tensor,  # (bs, response_length)
    loss_agg_mode: str,
    config,
    rollout_is_weights=None,
):
    # 1. 计算 importance ratio
    negative_approx_kl = log_prob - old_log_prob  # ← log_prob 需要梯度
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)  # ← ratio 需要梯度

    # 2. 计算 clipped surrogate objective
    pg_losses1 = -advantages * ratio  # ← 梯度从 ratio 传播
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)

    # 3. 取 max（PPO clip）
    pg_losses = torch.where(advantages < 0, ..., torch.maximum(pg_losses1, pg_losses2))

    # 4. 聚合为标量 loss
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    # → pg_loss: 标量 ← **需要梯度！**

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
```

**关键理解**：
- `log_prob` 来自 FSDP 模型的 forward，包含计算图
- `old_log_prob` 和 `advantages` 已经 detach，不参与梯度计算
- Loss 对 `log_prob` 求梯度，进而对模型参数求梯度

---

##### Backward Pass（FSDP 梯度计算）

```python
# 文件: verl/workers/actor/dp_actor.py:482-497

# ========== 梯度缩放（用于梯度累积）==========
if self.config.use_dynamic_bsz:
    loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
else:
    loss_scale_factor = 1 / self.gradient_accumulation

loss = policy_loss * loss_scale_factor

# ========== 关键：Backward Pass ==========
loss.backward()
# ↓ FSDP 自动执行以下操作（ZeRO-3 模式）：
#
# 1. **Backward All-Gather**：
#    - 逐层聚合完整参数（临时，用于计算梯度）
#    - 显存峰值：+16GB（临时）
#
# 2. **梯度计算**：
#    - 计算 loss 对每一层参数的梯度
#    - 使用 autograd 反向传播
#
# 3. **Reduce-Scatter 梯度**：
#    - 将完整梯度进行 All-Reduce 求和
#    - 然后 Scatter 到各个 GPU（每个 GPU 只保留自己分片的梯度）
#    - 梯度分片：2GB（bf16 格式）
#
# 4. **释放临时参数**：
#    - 释放 All-Gather 聚合的完整参数
#    - 只保留分片参数（2GB）和分片梯度（2GB）
#
# 显存变化：
# - Backward 开始: 6GB（分片参数 + 激活值）
# - Backward 中: 22GB（临时完整参数 + 激活值）
# - Backward 后: 4GB（分片参数 + 分片梯度）

# ========== 收集 metrics ==========
micro_batch_metrics.update({
    "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
    "actor/pg_clipfrac": pg_clipfrac.detach().item(),
    "actor/ppo_kl": ppo_kl.detach().item(),
    "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
})
```

**FSDP Backward 的详细机制**（ZeRO-3）：

```
┌────────────────────────────────────────────────────────────┐
│              FSDP Backward 逐层梯度计算流程                │
└────────────────────────────────────────────────────────────┘

假设模型有 32 层，每层参数 0.5GB（完整），在 8 个 GPU 上分片。

层 31（输出层）:
├── All-Gather 聚合完整参数（0.5GB）
├── 计算梯度: ∂loss/∂W_31
├── All-Reduce + Scatter 梯度（每个 GPU 保留 0.0625GB 分片梯度）
└── 释放临时参数

层 30:
├── All-Gather 聚合完整参数（0.5GB）
├── 计算梯度: ∂loss/∂W_30
├── All-Reduce + Scatter 梯度
└── 释放临时参数

...

层 0（输入层）:
├── All-Gather 聚合完整参数（0.5GB）
├── 计算梯度: ∂loss/∂W_0
├── All-Reduce + Scatter 梯度
└── 释放临时参数

┌────────────────────────────────────────────────────────────┐
│                    显存峰值分析                            │
└────────────────────────────────────────────────────────────┘

每一层 Backward 时:
├── 临时聚合参数: 0.5GB（当前层）
├── 已计算的梯度分片: 0.0625GB × (32 - 当前层数)
├── 激活值: ~4GB（取决于 batch size）
└── 总计: ~5-6GB（单层峰值）

全局峰值:
├── 如果启用 activation checkpointing: ~6GB
├── 如果未启用: ~10GB（需要保留所有激活值）
```

---

##### Optimizer Step（FSDP 参数更新）

```python
# 文件: verl/workers/actor/dp_actor.py:499-501

grad_norm = self._optimizer_step()
# ↓ 内部实现（verl/workers/actor/dp_actor.py:274-293）

def _optimizer_step(self):
    # ========== 步骤 1: 梯度裁剪 ==========
    if isinstance(self.actor_module, FSDP):  # FSDP1
        grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
    elif isinstance(self.actor_module, FSDPModule):  # FSDP2
        grad_norm = fsdp2_clip_grad_norm_(
            self.actor_module.parameters(),
            max_norm=self.config.grad_clip
        )
    # → FSDP 自动在梯度裁剪时进行 All-Reduce（计算全局梯度范数）

    # ========== 步骤 2: 检查梯度有效性 ==========
    if not torch.isfinite(grad_norm):
        print(f"WARN: grad_norm is not finite: {grad_norm}")
        self.actor_optimizer.zero_grad()  # 跳过此次更新
    else:
        # ========== 步骤 3: 优化器更新参数 ==========
        self.actor_optimizer.step()
        # ↓ AdamW 更新公式（ZeRO-3 模式，只更新本地分片）：
        #
        # 对于 GPU i 的参数分片 θ_i:
        # 1. 更新一阶动量: m_i = β1 * m_i + (1 - β1) * g_i
        # 2. 更新二阶动量: v_i = β2 * v_i + (1 - β2) * g_i²
        # 3. 更新参数: θ_i = θ_i - lr * m_i / (sqrt(v_i) + ε)
        #
        # 注意：每个 GPU 只更新自己的分片（2GB），不需要通信

    return grad_norm

# ========== 步骤 4: 清空梯度（准备下一个 micro-batch）==========
self.actor_optimizer.zero_grad()
```

---

#### 4.6.2 GRPO 更新阶段的显存时间线

```
┌────────────────────────────────────────────────────────────┐
│          单个 Micro-Batch 的显存变化（GPU 0）              │
└────────────────────────────────────────────────────────────┘

时刻 0ms: 准备阶段
├── Actor FSDP 分片: 2GB
├── Actor 优化器: 4GB
└── 总计: 6GB

时刻 0-10ms: Forward Pass
├── All-Gather 完整参数: +16GB (临时)
├── Forward 计算: +4GB (激活值)
├── 释放临时参数: -16GB
└── 总计: 6GB + 4GB = 10GB

时刻 10-30ms: Backward Pass（逐层）
├── 每层 All-Gather: +0.5GB (临时)
├── 梯度计算: +0.0625GB (分片梯度)
├── 释放临时参数: -0.5GB
├── 峰值（某一层）: 10GB + 0.5GB = 10.5GB
└── 结束后总计: 6GB + 2GB (分片梯度) = 8GB

时刻 30-35ms: Optimizer Step
├── 梯度裁剪: All-Reduce 梯度范数（通信，不增加显存）
├── 参数更新: in-place 更新分片参数（不增加显存）
└── 总计: 8GB

时刻 35ms: 清空梯度
├── zero_grad(): -2GB (分片梯度)
└── 总计: 6GB (回到初始状态)

┌────────────────────────────────────────────────────────────┐
│              完整 Mini-Batch 的显存变化                    │
└────────────────────────────────────────────────────────────┘

假设：
- Mini-Batch Size = 128
- Micro-Batch Size = 16
- Gradient Accumulation Steps = 8

梯度累积流程:
├── Micro-Batch 1: Forward + Backward → 累积梯度（1/8）
├── Micro-Batch 2: Forward + Backward → 累积梯度（2/8）
├── ...
├── Micro-Batch 8: Forward + Backward → 累积梯度（8/8）
└── Optimizer Step → 更新参数，清空梯度

显存峰值: ~10.5GB（Forward + 单层 Backward）
```

---

#### 4.6.3 哪些 Tensor 参与反向传播？

**总结表格**：

| Tensor | 形状 | 来源 | 需要梯度？ | 用途 |
|--------|------|------|-----------|------|
| **log_prob** | (bs, response_length) | **FSDP Forward 计算** | **✅ 是** | **当前策略的 log probabilities**<br>→ 反向传播到模型参数 |
| old_log_probs | (bs, response_length) | compute_log_prob() 阶段 | ❌ 否 | 旧策略的 log probabilities<br>→ 用于 importance ratio |
| advantages | (bs, response_length) | compute_advantage() 阶段（GRPO） | ❌ 否 | 优势估计<br>→ 权重，不参与梯度 |
| ref_log_prob | (bs, response_length) | compute_log_prob() 阶段（ref 模型） | ❌ 否 | 参考策略的 log probabilities<br>→ 用于 KL penalty |
| response_mask | (bs, response_length) | 输入数据 | ❌ 否 | Mask（哪些 token 有效） |
| entropy | (bs, response_length) | FSDP Forward 计算 | ✅ 是（可选） | Entropy bonus（可选） |

**梯度传播路径**：

```
Loss (标量)
  ↓ ∂loss/∂policy_loss
policy_loss
  ↓ ∂policy_loss/∂pg_loss
pg_loss
  ↓ ∂pg_loss/∂ratio
ratio = exp(log_prob - old_log_prob)
  ↓ ∂ratio/∂log_prob
log_prob (需要梯度)
  ↓ ∂log_prob/∂logits
logits (FSDP Forward 输出)
  ↓ ∂logits/∂hidden_states
hidden_states
  ↓ ∂hidden_states/∂W (模型参数)
FSDP 模型参数 W
```

**关键点**：
1. **只有 `log_prob` 需要梯度**（以及可选的 `entropy`）
2. `old_log_probs`、`advantages`、`ref_log_prob` 都使用 `.detach()` 或来自非梯度计算
3. **FSDP 在 Backward 时自动处理**：
   - All-Gather 聚合完整参数（临时）
   - 计算梯度
   - Reduce-Scatter 梯度分片
   - 释放临时参数

---

#### 4.6.4 GRPO 与 PPO 的区别

| 特性 | PPO（使用 Critic） | GRPO（Group Relative） |
|------|-------------------|----------------------|
| **Advantage 计算** | GAE（使用 Value Function） | Group-relative（组内对比） |
| **需要 Critic 模型** | ✅ 是 | ❌ 否 |
| **compute_advantage() 输入** | token_level_rewards, values, ... | token_level_rewards, index, ... |
| **Advantage 公式** | `A = δ + γλ A_{t+1}`<br>（δ = r + γV_{t+1} - V_t） | `A = (r - mean(r_group)) / std(r_group)` |
| **update_actor() 输入** | old_log_probs, advantages, values | old_log_probs, advantages |
| **Policy Loss** | Clipped surrogate objective | Clipped surrogate objective（相同） |
| **需要梯度的 Tensor** | log_prob, entropy | log_prob, entropy（相同） |

**GRPO 的优势**：
1. **不需要 Critic 模型** → 节省 50% 训练资源
2. **Advantage 计算更简单** → 无需 Value Function 估计
3. **适合 Outcome Supervision** → 只需要序列级 reward

---

---

## 5. 模式切换的详细机制

### 5.1 `rollout_mode()` - 从 Trainer 切换到 Rollout

**代码位置**：`verl/workers/fsdp_workers.py:658-739`

**调用时机**：每次 `generate_sequences()` 开始时

```python
# 文件: verl/workers/fsdp_workers.py:658-739

async def rollout_mode(self):
    """Context switch hybridengine to rollout mode."""

    # ========== 子步骤 1: 清理缓存 ==========
    aggressive_empty_cache(force_sync=True)
    # → torch.cuda.empty_cache()
    # → 清理碎片化的显存

    # ========== 子步骤 2: 加载 Actor FSDP 到 GPU（如果之前 offload 了）==========
    log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
    if self._is_offload_param:
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
        # → 从 CPU 加载参数分片到 GPU
        # → 显存增加 2GB
    log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

    # ========== 子步骤 3: 收集 FSDP 参数（分片 → 完整）==========
    peft_config = None
    peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)

    if hasattr(peft_model, "peft_config"):  # LoRA 模式
        # 仅收集 LoRA 参数（轻量级）
        params = collect_lora_params(
            module=self.actor_module_fsdp,
            layered_summon=self.config.rollout.get("layered_summon", False),
            base_sync_done=self.base_sync_done,
        )
    else:  # 完整模型
        # 收集完整参数（重量级）
        params = self.actor_module_fsdp.state_dict()
        # → 内部调用 FSDP 的 state_dict()
        # → 根据配置决定是 SHARDED_STATE_DICT 还是 FULL_STATE_DICT

    # ========== 子步骤 4: 转换参数格式 ==========
    params = convert_weight_keys(
        params,
        getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
    )
    # → 转换 FSDP 的 key 格式为 Rollout 期望的格式
    # → 例如：移除 "_fsdp_wrapped_module." 前缀

    # ========== 子步骤 5: 转换为 full_tensor（关键！）==========
    # 对于 FSDP2 使用 DTensor，需要显式转换
    if fsdp_version(self.actor_module_fsdp) == 2:  # FSDP2
        device = get_device_id()
        per_tensor_param = (
            (name, param.to(device, non_blocking=True).full_tensor()
             if isinstance(param, DTensor) else param)
            for name, param in params.items()
        )
    else:  # FSDP1
        per_tensor_param = params.items()

    # DTensor.full_tensor() 的作用：
    # 1. 触发 all-gather 操作（NCCL 通信）
    # 2. 将 8 个 GPU 的分片聚合到每个 GPU
    # 3. 返回完整的 tensor（16GB）
    #
    # 通信量：
    # - GPU 0 发送 2GB，接收 14GB → 总计 16GB
    # - GPU 1 发送 2GB，接收 14GB → 总计 16GB
    # - ...
    # - 总通信量：8 * 14GB = 112GB
    # - 通信时间：~200ms（NVLink 带宽 600GB/s）

    # ========== 子步骤 6: 恢复 Rollout Engine（重新加载 KV Cache）==========
    if self.config.rollout.free_cache_engine:
        await self.rollout.resume(tags=["weights"])
        # → 重新分配 KV Cache 空间（32GB）
        # → 显存增加 32GB
    log_gpu_memory_usage("After resume weights", logger=logger)

    # ========== 子步骤 7: 更新 Rollout 参数（核心！）==========
    # 文件: verl/workers/fsdp_workers.py:725-728
    await self.rollout.update_weights(per_tensor_param, peft_config=peft_config)
    # → 将 Actor FSDP 的参数复制到 Rollout 模型
    # → in-place 更新，不需要额外显存
    # → 耗时：~100ms（复制 16GB 数据）

    log_gpu_memory_usage("After update_weights", logger=logger)
    del params, per_tensor_param
    aggressive_empty_cache(force_sync=True)

    # ========== 子步骤 8: 恢复 KV Cache ==========
    if self.config.rollout.free_cache_engine:
        await self.rollout.resume(tags=["kv_cache"])
        # → 重新分配 KV Cache（32GB）
    log_gpu_memory_usage("After resume kv_cache", logger=logger)

    # ========== 子步骤 9: 保存/恢复随机状态 ==========
    self.base_sync_done = True
    # 保存 Trainer 的随机状态
    self.torch_random_states = get_torch_device().get_rng_state()
    # 恢复 Rollout 的随机状态
    get_torch_device().set_rng_state(self.gen_random_states)
    # → 确保生成的随机性可复现
```

**显存变化时间线**：

```
初始状态（Trainer Mode）:
├── Actor FSDP 分片: 2GB (GPU)
├── Actor 优化器: 4GB (GPU 或 CPU offload)
├── Rollout 参数: 16GB (GPU)
└── 总计: 18-22GB

执行 rollout_mode():
├── load_fsdp_model_to_gpu: +2GB (如果之前 offload)
├── params.full_tensor(): +16GB (临时，all-gather)
├── update_weights: 0GB (in-place 替换)
├── 释放临时参数: -16GB
├── resume KV Cache: +32GB
└── 总计: 50GB

最终状态（Rollout Mode）:
├── Actor FSDP 分片: 2GB (GPU 或 CPU offload)
├── Rollout 参数: 16GB (GPU, 已更新)
├── Rollout KV Cache: 32GB (GPU)
└── 总计: 50GB (如果 Actor offload 则 48GB)
```

### 5.2 `trainer_mode()` - 从 Rollout 切换回 Trainer

**代码位置**：`verl/workers/fsdp_workers.py:741-757`

**调用时机**：每次 `generate_sequences()` 结束时

```python
# 文件: verl/workers/fsdp_workers.py:741-757

async def trainer_mode(self):
    """Context switch hybridengine to trainer mode."""

    # ========== 子步骤 1: 释放 Rollout KV Cache（关键！）==========
    if self.config.rollout.free_cache_engine:
        log_gpu_memory_usage("Before rollout offload", logger=logger)
        await self.rollout.release()
        # → 释放 KV Cache（32GB）
        # → 保留 Rollout 参数（16GB）
        log_gpu_memory_usage("After rollout offload", logger=logger)

    # rollout.release() 内部逻辑：
    # async def release(self):
    #     """释放 KV Cache 和其他缓存"""
    #     # 1. 释放所有 KV Cache 块
    #     self.kv_cache_manager.clear()  # → 释放 32GB
    #
    #     # 2. 清理其他缓存（如 PageTable）
    #     self.cache_engine.clear()
    #
    #     # 3. 保留模型参数（不释放）
    #     # self.model.parameters() 仍然在 GPU（16GB）
    #
    #     torch.cuda.empty_cache()

    # ========== 子步骤 2: 切换 Actor 模型为训练模式 ==========
    self.actor_module_fsdp.train()
    # → 启用 Dropout、BatchNorm 等训练特性

    # ========== 子步骤 3: 清理缓存 ==========
    aggressive_empty_cache(force_sync=True)
    # → torch.cuda.empty_cache()
    # → 整理碎片化的显存

    # ========== 子步骤 4: 设置可扩展段（PyTorch 显存管理优化）==========
    set_expandable_segments(True)
    # → 允许 PyTorch 动态扩展显存段
    # → 减少显存碎片化

    # ========== 子步骤 5: 恢复随机状态 ==========
    # 保存 Rollout 的随机状态
    self.gen_random_states = get_torch_device().get_rng_state()
    # 恢复 Trainer 的随机状态
    get_torch_device().set_rng_state(self.torch_random_states)
    # → 确保训练的随机性可复现
```

**显存变化时间线**：

```
初始状态（Rollout Mode）:
├── Actor FSDP 分片: 2GB (可能在 CPU)
├── Rollout 参数: 16GB (GPU)
├── Rollout KV Cache: 32GB (GPU)
└── 总计: 50GB

执行 trainer_mode():
├── rollout.release(): -32GB (释放 KV Cache)
├── empty_cache(): 0GB (整理碎片)
└── 总计: 18GB

最终状态（Trainer Mode）:
├── Actor FSDP 分片: 2GB (GPU)
├── Actor 优化器: 4GB (GPU 或 CPU offload)
├── Rollout 参数: 16GB (GPU, 保留但不使用)
└── 总计: 18-22GB
```

---

## 6. 参数同步的底层实现

### 6.1 为什么每次 Rollout 都需要同步参数？

因为 Actor 模型在每次训练后都会更新！

```python
# ========== 时间线 ==========
时刻 t0: 初始化
  Actor FSDP 参数 = W0
  Rollout 参数 = W0
  （两者一致）

时刻 t1: 第一次训练
  generate_sequences():
    rollout_mode() → Actor (W0) → Rollout (W0)  # 同步（实际上参数已一致，但仍执行）
    Rollout 推理（使用 W0）
  update_actor():
    训练更新 → Actor (W1)  # 参数已改变！

时刻 t2: 第二次训练
  generate_sequences():
    rollout_mode() → Actor (W1) → Rollout (W1)  # 同步（必须！Rollout 还是 W0）
    Rollout 推理（使用 W1）  # 使用最新的策略
  update_actor():
    训练更新 → Actor (W2)

时刻 t3: 第三次训练
  generate_sequences():
    rollout_mode() → Actor (W2) → Rollout (W2)  # 同步
    Rollout 推理（使用 W2）
  ...
```

**如果不同步会怎样？**

```python
# 错误示例：不同步
时刻 t1: Actor = W0, Rollout = W0 → 生成使用 W0 ✓
时刻 t2: Actor = W1, Rollout = W0 → 生成使用 W0 ✗ (应该用 W1!)
时刻 t3: Actor = W2, Rollout = W0 → 生成使用 W0 ✗ (应该用 W2!)

结果：
- Rollout 使用的是初始策略，不是当前策略
- PPO 的 importance sampling 失效
- 训练发散！
```

### 6.2 FSDP 分片与参数状态

**关键问题**：FSDP 初始化后，每个 GPU 上的 `actor_module_fsdp` 是否包含完整的模型参数？

**答案**：**不是！每个 GPU 只包含部分参数（分片）**

#### 6.2.1 FSDP 分片原理（ZeRO-3）

```python
# ========== FSDP 初始化后的参数分布 ==========

# 模型: Qwen3-8B (8B 参数, bf16 格式 = 16GB)
# GPU 数量: 8 个 GPU
# FSDP 策略: FULL_SHARD (ZeRO-3)

# 每个 GPU 上的参数分片:
GPU 0: 分片 0 (2GB, 参数 0-1B)
GPU 1: 分片 1 (2GB, 参数 1-2B)
GPU 2: 分片 2 (2GB, 参数 2-3B)
GPU 3: 分片 3 (2GB, 参数 3-4B)
GPU 4: 分片 4 (2GB, 参数 4-5B)
GPU 5: 分片 5 (2GB, 参数 5-6B)
GPU 6: 分片 6 (2GB, 参数 6-7B)
GPU 7: 分片 7 (2GB, 参数 7-8B)

# 对于单个 GPU 进程而言:
# - actor_module_fsdp 只包含本地分片（2GB）
# - 无法直接用于推理（参数不完整）
# - 训练时通过 All-Gather 临时聚合完整参数
```

**FSDP 分片的内部表示**：

```python
# FSDP1 (PyTorch < 2.4):
# - 参数存储为普通 torch.Tensor
# - 通过 FSDP 的 _summon_full_params() 聚合

# FSDP2 (PyTorch >= 2.4):
# - 参数存储为 DTensor（分布式 Tensor）
# - DTensor 自动追踪分片信息和设备映射
# - 通过 DTensor.full_tensor() 触发 All-Gather

# 示例: 一个权重矩阵 [8192, 8192]
# FSDP2 中的存储:
GPU 0: DTensor(local_shard=[1024, 8192], device_mesh=DeviceMesh([0,1,...,7]))
GPU 1: DTensor(local_shard=[1024, 8192], device_mesh=DeviceMesh([0,1,...,7]))
...
GPU 7: DTensor(local_shard=[1024, 8192], device_mesh=DeviceMesh([0,1,...,7]))

# 调用 full_tensor() 后:
# → 返回完整的 Tensor [8192, 8192]
# → 内部执行 NCCL All-Gather 聚合所有分片
```

---

### 6.3 FSDP state_dict() 的机制

**代码位置**：
- 配置 state_dict 类型：`verl/workers/fsdp_workers.py:634-644`
- 调用 state_dict()：`verl/workers/fsdp_workers.py:679`

```python
# ========== 步骤 1: 配置 state_dict 类型 ==========
# 文件: verl/workers/fsdp_workers.py:634-644

if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:
    FSDP.set_state_dict_type(
        self.actor_module_fsdp,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(),
    )
elif fsdp_version(self.actor_module_fsdp) == 1:
    FSDP.set_state_dict_type(
        self.actor_module_fsdp,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,  # ← 多 GPU 使用分片模式
        state_dict_config=ShardedStateDictConfig(),
    )


# ========== 步骤 2: 调用 state_dict() ==========
# 文件: verl/workers/fsdp_workers.py:679

params = self.actor_module_fsdp.state_dict()

# FSDP state_dict() 的内部行为（根据配置）：

# 1. 如果配置为 SHARDED_STATE_DICT：
#    - 每个 GPU 只返回本地分片（2GB）
#    - 不需要 all-gather
#    - 显存开销：0GB（仅返回已有分片的引用）
#    - 返回的 params 是分片的（DTensor）

# 2. 如果配置为 FULL_STATE_DICT：
#    - 内部调用 FSDP.summon_full_params()
#    - All-Gather 聚合所有分片到每个 GPU
#    - 临时显存开销：16GB（完整参数）
#    - 返回后立即释放临时参数


# ========== verl 使用 SHARDED_STATE_DICT 的原因 ==========
# 1. 节省显存：不在 state_dict() 阶段触发 all-gather
# 2. 延迟聚合：在后续的 full_tensor() 调用时才触发 all-gather
# 3. 灵活性：可以选择性地聚合某些层（layered_summon）
```

#### 6.3.1 verl 使用 SHARDED_STATE_DICT 的策略

**为什么 verl 选择 SHARDED_STATE_DICT 而不是 FULL_STATE_DICT？**

1. **节省显存**：
   - `FULL_STATE_DICT`: 在 `state_dict()` 时就触发 All-Gather（峰值 +16GB）
   - `SHARDED_STATE_DICT`: 延迟到 `full_tensor()` 调用时才 All-Gather

2. **灵活性**：
   - 可以选择性地聚合某些层（Layered Summon）
   - 逐层聚合可以显著降低显存峰值（从 16GB 降到 ~0.5GB）

3. **兼容性**：
   - FSDP2 原生使用 DTensor，自动支持分片
   - FSDP1 也可以配置为 SHARDED_STATE_DICT

---

### 6.4 DTensor.full_tensor() - All-Gather 的触发点

**代码位置**：`verl/workers/fsdp_workers.py:708-714`

```python
# 文件: verl/workers/fsdp_workers.py:708-714

if fsdp_version(self.actor_module_fsdp) == 2:  # FSDP2
    device = get_device_id()
    per_tensor_param = (
        (name, param.to(device, non_blocking=True).full_tensor()
         if isinstance(param, DTensor) else param)  # ← 这里触发 all-gather
        for name, param in params.items()
    )
else:  # FSDP1
    per_tensor_param = params.items()


# ========== DTensor.full_tensor() 的作用 ==========
# DTensor 是分片的 tensor，每个 GPU 只有一部分

# 例如：一个 [8192, 8192] 的权重矩阵
# - GPU 0: DTensor，本地分片 [1024, 8192] (2GB / 8)
# - GPU 1: DTensor，本地分片 [1024, 8192]
# - ...
# - GPU 7: DTensor，本地分片 [1024, 8192]

# 调用 full_tensor() 后：
# 1. 触发 NCCL all-gather 操作
# 2. 每个 GPU 收集所有分片
# 3. 返回完整的 tensor [8192, 8192] (16GB / 模型层数)

# 通信模式（All-Gather）：
#     GPU 0       GPU 1       GPU 2   ...   GPU 7
#       ↓           ↓           ↓             ↓
#    [shard0]    [shard1]    [shard2]      [shard7]
#       ↓           ↓           ↓             ↓
#    ┌───────────────────────────────────────────┐
#    │         NCCL All-Gather Ring              │
#    │   每个 GPU 广播自己的分片到所有 GPU        │
#    └───────────────────────────────────────────┘
#       ↓           ↓           ↓             ↓
#    [full]      [full]      [full]        [full]
#    16GB        16GB        16GB          16GB

# 通信量计算：
# - 每个 GPU 发送：2GB (自己的分片)
# - 每个 GPU 接收：14GB (其他 7 个 GPU 的分片)
# - 总通信量（per GPU）：2GB + 14GB = 16GB
# - 全局总通信量：8 * 14GB = 112GB
# - 通信时间：~200ms（NVLink 带宽 600GB/s）
```

### 6.4 rollout.update_weights() - In-Place 参数更新

**代码位置**：`verl/workers/fsdp_workers.py:728`

```python
# 文件: verl/workers/fsdp_workers.py:728

await self.rollout.update_weights(per_tensor_param, peft_config=peft_config)

# ========== rollout.update_weights() 内部（vLLM/SGLang）==========
# 伪代码（基于 vLLM 实现）：

async def update_weights(self, per_tensor_param, peft_config=None):
    """更新 Rollout 模型的参数"""

    # 1. 遍历所有参数
    for name, new_param in per_tensor_param:
        # new_param: [完整的 8B 参数中的某一层]
        # 例如: 'model.layers.0.self_attn.q_proj.weight' -> [4096, 4096] (32MB)

        # 2. 获取 Rollout 模型中对应的参数
        old_param = self.model.get_parameter(name)  # 也是 [4096, 4096]

        # 3. 直接替换（in-place 更新）
        old_param.data.copy_(new_param.data)
        # → GPU 上的内存复制（cudaMemcpy）
        # → 不需要额外显存（in-place）
        # → 耗时：~100ms（复制 16GB 数据）

    # 4. 更新完成后，释放临时参数
    del per_tensor_param
    torch.cuda.empty_cache()


# ========== 为什么是 in-place 更新？ ==========
# 1. 节省显存：不需要分配新的 16GB 空间
# 2. 高效：直接在 GPU 上复制，无需 CPU 中转
# 3. 保持 Rollout Engine 状态：KV Cache、PageTable 等不受影响
```

```python
# 文件: verl/workers/fsdp_workers.py:708-714

if fsdp_version(self.actor_module_fsdp) == 2:  # FSDP2
    device = get_device_id()
    per_tensor_param = (
        (name, param.to(device, non_blocking=True).full_tensor()
         if isinstance(param, DTensor) else param)  # ← 这里触发 all-gather
        for name, param in params.items()
    )
else:  # FSDP1
    per_tensor_param = params.items()


# ========== DTensor.full_tensor() 的作用 ==========
# DTensor 是分片的 tensor，每个 GPU 只有一部分

# 例如：一个 [8192, 8192] 的权重矩阵
# - GPU 0: DTensor，本地分片 [1024, 8192] (2GB / 8)
# - GPU 1: DTensor，本地分片 [1024, 8192]
# - ...
# - GPU 7: DTensor，本地分片 [1024, 8192]

# 调用 full_tensor() 后：
# 1. 触发 NCCL all-gather 操作
# 2. 每个 GPU 收集所有分片
# 3. 返回完整的 tensor [8192, 8192] (16GB / 模型层数)

# 通信模式（All-Gather）：
#     GPU 0       GPU 1       GPU 2   ...   GPU 7
#       ↓           ↓           ↓             ↓
#    [shard0]    [shard1]    [shard2]      [shard7]
#       ↓           ↓           ↓             ↓
#    ┌───────────────────────────────────────────┐
#    │         NCCL All-Gather Ring              │
#    │   每个 GPU 广播自己的分片到所有 GPU        │
#    └───────────────────────────────────────────┘
#       ↓           ↓           ↓             ↓
#    [full]      [full]      [full]        [full]
#    16GB        16GB        16GB          16GB

# 通信量计算：
# - 每个 GPU 发送：2GB (自己的分片)
# - 每个 GPU 接收：14GB (其他 7 个 GPU 的分片)
# - 总通信量（per GPU）：2GB + 14GB = 16GB
# - 全局总通信量：8 * 14GB = 112GB
# - 通信时间：~200ms（NVLink 带宽 600GB/s）
```

**All-Gather 的显存影响**：

```
初始状态（rollout_mode 开始）:
├── Actor FSDP 分片: 2GB (本地分片)
├── Actor 优化器: 4GB (或 CPU offload)
├── Rollout 参数: 16GB
└── 总计: 18-22GB

执行 params.full_tensor():
├── 原始分片: 2GB (保留)
├── 聚合完整参数: +16GB (临时)
└── 总计: 22GB + 16GB = 38GB ← 这是临时峰值

完成后:
├── 释放临时参数: -16GB
├── per_tensor_param 生成器: 引用聚合结果
└── 总计: 回到 22GB
```

---

### 6.5 rollout.update_weights() - In-Place 参数更新

**代码位置**：`verl/workers/fsdp_workers.py:728`

```python
# 文件: verl/workers/fsdp_workers.py:728

await self.rollout.update_weights(per_tensor_param, peft_config=peft_config)
```

#### 6.5.1 vLLM update_weights() 实现

**代码位置**：`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:430-453`

```python
# 文件: verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:430-453

async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
    """Update the weights of the rollout model.

    Args:
        weights: A generator that yields the name of the weight tensor and the tensor itself.
    """
    peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)

    if peft_config and base_sync_done:
        # ========== LoRA 模式：只更新 LoRA 参数 ==========
        lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
        lora_reqest = TensorLoRARequest(
            lora_name=f"{lora_int_id}",
            lora_int_id=lora_int_id,
            lora_path="simon_lora_path",
            peft_config=asdict(peft_config),
            lora_tensors=weights,  # LoRA 参数（小，例如 100MB）
        )
        self.inference_engine.llm_engine.add_lora(lora_reqest)
        logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
    else:
        # ========== 完整模型模式：更新所有参数 ==========
        from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

        # 获取 vLLM 的模型对象
        model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model

        # 针对 MoE 模型打补丁（如果需要）
        patch_vllm_moe_model_weight_loader(model)

        # 调用 vLLM 的 load_weights() 方法
        model.load_weights(weights)
        # ↓ vLLM 内部实现（伪代码）:
        # def load_weights(self, weights_generator):
        #     for name, param_tensor in weights_generator:
        #         # 1. 找到模型中对应的参数
        #         model_param = self.get_parameter(name)
        #
        #         # 2. in-place 复制（关键！）
        #         model_param.data.copy_(param_tensor.data)
        #         # → GPU 上的内存复制（cudaMemcpy）
        #         # → 不需要额外显存（in-place）
        #         # → 耗时：~100ms（复制 16GB 数据）


# ========== 为什么是 in-place 更新？ ==========
# 1. 节省显存：不需要分配新的 16GB 空间
# 2. 高效：直接在 GPU 上复制，无需 CPU 中转
# 3. 保持 Rollout Engine 状态：KV Cache、PageTable 等不受影响
```

**vLLM 参数更新的内存布局**：

```
更新前（Rollout Mode）:
GPU 显存:
├── Rollout 参数（旧）: 16GB [W_old]
├── KV Cache: 32GB
└── 总计: 48GB

更新中（执行 model.load_weights）:
GPU 显存:
├── Rollout 参数（被覆盖）: 16GB [W_old → W_new]
│   ├── model_param.data.copy_(param_tensor.data)
│   ├── in-place 替换，逐层复制
│   └── 不需要额外显存！
├── KV Cache: 32GB（不受影响）
└── 总计: 48GB（显存不变）

更新后:
GPU 显存:
├── Rollout 参数（新）: 16GB [W_new]
├── KV Cache: 32GB
└── 总计: 48GB
```

---

#### 6.5.2 SGLang update_weights() 实现

**代码位置**：`verl/workers/rollout/sglang_rollout/sglang_rollout.py:1505-1528`

```python
# 文件: verl/workers/rollout/sglang_rollout/sglang_rollout.py:1505-1528

async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
    """
    Update model weights using tensor buckets, similar to THUDM/slime's implementation.

    Notes:
      - For the best performance of `rebuild_cuda_tensor`, it is recommended to:
          1. Enable `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`.
          2. Manually set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
        when using Tensor Parallelism (TP >= 8).
      - See reference implementations in SLIME:
        - Main logic: https://github.com/THUDM/slime/blob/fb7605cc/slime/ray/ppo_actor.py#L452
        - runtime envs: https://github.com/THUDM/slime/blob/fb7605cc/slime/ray/ppo_actor.py#L39
    """
    # ========== SGLang 的分批更新策略 ==========
    # 将大参数分为多个小批次（buckets），逐批传输和更新
    # 默认 bucket 大小：256MB
    update_weights_bucket_bytes = int(self.config.update_weights_bucket_megabytes) << 20

    # get_named_tensor_buckets: 将 weights 生成器分割为多个 bucket
    # 例如: 16GB 参数 → 64 个 256MB buckets
    for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
        # 调用 SGLang 的权重更新接口
        await sgl_update_weights(
            engine=self._engine,
            params_batch=params_batch,  # 当前 bucket 的参数
            device_mesh_key="infer_tp",
            device_mesh=self.device_mesh,
        )

    # sgl_update_weights 内部实现（简化）:
    # 1. 将参数从 Ray object store 传输到 SGLang Engine
    # 2. 如果是 TP（Tensor Parallelism）模式，广播到所有 TP ranks
    # 3. 调用 engine.update_weights_from_tensor() 执行 in-place 更新
    # 4. 类似 vLLM，使用 param.data.copy_()

    # 清空 Radix Cache（前缀缓存）
    if self.device_mesh["infer_tp"].get_local_rank() == 0:
        await self._engine.flush_cache()
        # 清空前缀缓存，避免使用旧参数生成的 KV Cache


# ========== SGLang 与 vLLM 的关键区别 ==========
# 1. 分批策略:
#    - vLLM: 一次性传输所有参数（16GB 生成器）
#    - SGLang: 分批传输（例如 64 个 256MB buckets）
#
# 2. 优势:
#    - SGLang 的分批策略减少了 Ray object store 的压力
#    - 适合超大模型（100B+），避免单次传输过大
#
# 3. 缺点:
#    - SGLang 需要多次异步调用（64 次 vs 1 次）
#    - 总耗时可能略长于 vLLM（但更稳定）
#
# 4. 实际性能:
#    - vLLM: ~100ms（16GB 一次性复制）
#    - SGLang: ~120ms（64 次分批复制 + 通信开销）
#    - 差异不大，都是 in-place 更新
```

**SGLang 分批更新的内存布局**：

```
更新前:
GPU 显存:
├── SGLang 参数（旧）: 16GB [W_old]
├── KV Cache: 32GB
└── 总计: 48GB

更新中（分批，例如 64 次）:
迭代 1: 更新前 256MB
├── 参数 [0:256MB]: W_old → W_new
└── 显存: 48GB（不变）

迭代 2: 更新 [256MB:512MB]
├── 参数 [256MB:512MB]: W_old → W_new
└── 显存: 48GB（不变）

...

迭代 64: 更新 [15.75GB:16GB]
├── 参数 [15.75GB:16GB]: W_old → W_new
└── 显存: 48GB（不变）

更新后:
GPU 显存:
├── SGLang 参数（新）: 16GB [W_new]
├── KV Cache: 32GB
└── 总计: 48GB
```

---

#### 6.5.3 vLLM vs SGLang 参数更新对比

| 特性 | vLLM | SGLang |
|------|------|--------|
| **更新策略** | 一次性传输所有参数 | 分批传输（buckets） |
| **单次调用** | 1 次 `model.load_weights()` | N 次 `sgl_update_weights()` |
| **Bucket 大小** | 完整 16GB | 可配置（默认 256MB） |
| **通信次数** | 1 次（大对象传输） | 64 次（小对象传输） |
| **Ray object store 压力** | 高（16GB 单个对象） | 低（256MB 小对象） |
| **适用场景** | 中等模型（7B-70B） | 超大模型（100B+） |
| **总耗时** | ~100ms | ~120ms |
| **显存开销** | In-place（0GB 额外） | In-place（0GB 额外） |
| **稳定性** | 可能触发 Ray OOM | 更稳定 |

**共同点**：
1. 都使用 `param.data.copy_()` in-place 更新
2. 都不需要额外显存（直接覆盖旧参数）
3. 都保持 Rollout Engine 的其他状态（KV Cache 不受影响）

**选择建议**：
- **7B-70B 模型**：使用 vLLM（更简单，性能更好）
- **100B+ 模型**：使用 SGLang（分批更新更稳定）
- **Ray 内存受限**：使用 SGLang（避免大对象传输）

---

### 6.6 trainer_mode() - 为什么只需要 train()?

**代码位置**：`verl/workers/fsdp_workers.py:741-757`

```python
# 文件: verl/workers/fsdp_workers.py:741-757

async def trainer_mode(self):
    """Context switch hybridengine to trainer mode."""

    # ========== 子步骤 1: 释放 Rollout KV Cache（关键！）==========
    if self.config.rollout.free_cache_engine:
        log_gpu_memory_usage("Before rollout offload", logger=logger)
        await self.rollout.release()
        # → 释放 KV Cache（32GB）
        # → 保留 Rollout 参数（16GB）
        log_gpu_memory_usage("After rollout offload", logger=logger)

    # ========== 子步骤 2: 切换 Actor 模型为训练模式 ==========
    self.actor_module_fsdp.train()
    # ↓ 这一步做了什么？

    # PyTorch nn.Module.train() 的作用：
    # 1. 设置 self.training = True
    # 2. 递归调用所有子模块的 train()
    # 3. 启用训练特性：
    #    - Dropout: 在训练时随机丢弃神经元（inference 时不丢弃）
    #    - BatchNorm: 更新 running_mean 和 running_var（inference 时使用固定值）
    #    - LayerNorm: 无影响（train 和 eval 行为一致）

    # 重要：train() 不涉及参数的加载或卸载！
    # - Actor FSDP 的参数分片（2GB）始终在 GPU（或 CPU 如果 offload）
    # - Actor 优化器状态（4GB）始终在 GPU（或 CPU 如果 offload）
    # - 这些权重和内容在整个训练过程中都保留在资源中

    # ========== 子步骤 3: 清理缓存 ==========
    aggressive_empty_cache(force_sync=True)
    # → torch.cuda.empty_cache()
    # → 整理碎片化的显存

    # ========== 子步骤 4: 设置可扩展段（PyTorch 显存管理优化）==========
    set_expandable_segments(True)
    # → 允许 PyTorch 动态扩展显存段
    # → 减少显存碎片化

    # ========== 子步骤 5: 恢复随机状态 ==========
    # 保存 Rollout 的随机状态
    self.gen_random_states = get_torch_device().get_rng_state()
    # 恢复 Trainer 的随机状态
    get_torch_device().set_rng_state(self.torch_random_states)
    # → 确保训练的随机性可复现
```

**为什么 trainer_mode() 看起来这么简单？**

**答案**：因为所有训练需要的状态都已经在内存中了！

```
Trainer Mode 的资源状态:
├── Actor FSDP 参数分片: 2GB (GPU 或 CPU offload)
│   └── 始终保留，从未释放
│
├── Actor 优化器状态: 4GB (GPU 或 CPU offload)
│   ├── momentum: 2GB
│   ├── variance: 2GB
│   └── 始终保留，从未释放
│
├── Rollout 参数: 16GB (GPU)
│   └── 保留但不使用（只在 Rollout Mode 使用）
│
├── Rollout KV Cache: 0GB (已释放)
│   └── trainer_mode() 中调用 release() 释放
│
└── 总计: 18-22GB（取决于 offload 配置）
```

**关键理解**：

1. **参数不需要重新加载**：
   - Actor FSDP 的参数分片一直在内存（GPU 或 CPU）
   - update_actor() 更新的就是这些参数
   - 下一次 rollout_mode() 会同步这些更新后的参数到 Rollout

2. **优化器状态不需要重新加载**：
   - AdamW 的 momentum 和 variance 一直在内存
   - 每次 update_actor() 会更新优化器状态

3. **train() 只是模式切换**：
   - 启用 Dropout（训练时随机失活）
   - 启用 BatchNorm 的统计量更新
   - 不涉及任何参数的移动或加载

4. **为什么 Rollout 参数还在 GPU？**：
   - Rollout 参数（16GB）保留在 GPU，但在 Trainer Mode 不使用
   - 这是为了避免频繁的参数加载/卸载（开销大）
   - 只释放 KV Cache（32GB），因为它占用最多且训练时不需要

**完整的状态对比**：

```
┌────────────────────────────────────────────────────────────┐
│                  Trainer Mode vs Rollout Mode              │
└────────────────────────────────────────────────────────────┘

Trainer Mode (update_actor, compute_log_prob):
├── Actor FSDP 分片: 2GB (使用中，在 GPU)
├── Actor 优化器: 4GB (使用中，在 GPU)
├── Rollout 参数: 16GB (不使用，在 GPU)
├── Rollout KV Cache: 0GB (已释放)
├── self.actor_module_fsdp.training = True
└── 总计: 22GB

Rollout Mode (generate_sequences):
├── Actor FSDP 分片: 2GB (不使用，可 offload 到 CPU)
├── Actor 优化器: 4GB (不使用，可 offload 到 CPU)
├── Rollout 参数: 16GB (使用中，已更新为最新)
├── Rollout KV Cache: 32GB (已分配，使用中)
├── self.actor_module_fsdp.training = False (通过 rollout_mode 自动设置)
└── 总计: 48-54GB（取决于 offload）
```

**总结**：

`trainer_mode()` 只需要 `train()` 这么简单，是因为：
1. 所有训练权重（FSDP 分片 + 优化器）始终在内存
2. `train()` 只是切换模块的行为模式（Dropout/BatchNorm）
3. 实际的资源释放只发生在 KV Cache（`rollout.release()`）
4. Rollout 参数保留在 GPU，避免频繁加载/卸载的开销

---

### 6.7 优化：分层同步（Layered Summon）

**代码位置**：`verl/utils/fsdp_utils.py:569-608`

对于大模型，可以逐层同步参数，减少显存峰值：

```python
# 文件: verl/utils/fsdp_utils.py:569-608

def layered_summon_lora_params(fsdp_module) -> OrderedDict:
    """逐层聚合参数，减少显存峰值"""
    lora_params = OrderedDict()

    # 遍历每一层（如 32 层 Transformer）
    prefix_list = [
        "_fsdp_wrapped_module.base_model.model.model.layers.",
        # ...
    ]

    for prefix in prefix_list:
        for name, submodule in __prefix_submodules(fsdp_module, prefix):
            # 只聚合当前层的参数
            if fsdp_version(submodule) > 0:
                with FSDP.summon_full_params(submodule, writeback=False):
                    sub_lora_params = get_peft_model_state_dict(peft_model, state_dict=submodule.state_dict())
                    sub_lora_params = {
                        f"{prefix}.{name}": param.full_tensor().detach().cpu()
                        if hasattr(param, "full_tensor")
                        else param.detach().cpu()
                        for name, param in sub_lora_params.items()
                    }
                    lora_params.update(sub_lora_params)
                    submodule._is_root = False
                # → 显存峰值：16GB / 32 = 0.5GB（单层）

            # 当前层处理完后立即释放
            get_torch_device().empty_cache()
            # → 显存立即恢复

    return lora_params

# 优势：
# - 不分层：峰值 16GB（所有层一次性聚合）
# - 分层：峰值 0.5GB（每次只聚合一层）
# - 代价：通信次数增加（32 次 vs 1 次）
```

---

## 7. 显存管理时间线

```
┌────────────────────────────────────────────────────────────────────┐
│               PPO 单次迭代的显存变化（GPU 0）                      │
└────────────────────────────────────────────────────────────────────┘

时刻 0s: Trainer Mode (初始)
├── Actor FSDP 分片: 2GB
├── Actor 优化器: 4GB (或 CPU offload: 0GB)
├── Rollout 参数: 16GB
└── 总计: 22GB (或 18GB)

时刻 0-1s: generate_sequences() → rollout_mode()
├── load_fsdp_model_to_gpu: 22GB → 24GB (+2GB, 如果之前 offload)
├── params.full_tensor(): 24GB → 40GB (+16GB, 临时)
├── update_weights: 40GB → 40GB (in-place)
├── 释放临时参数: 40GB → 24GB (-16GB)
├── resume KV Cache: 24GB → 56GB (+32GB)
└── 总计: 56GB ← 全局峰值

时刻 1-2s: Rollout 推理
├── KV Cache 动态使用: 56GB → 50GB (部分使用)
└── 总计: ~50GB

时刻 2s: trainer_mode()
├── release KV Cache: 50GB → 18GB (-32GB)
└── 总计: 18GB

时刻 2-3s: compute_log_prob()
├── load_fsdp_model_to_gpu: 18GB → 20GB (+2GB, 如果之前 offload)
├── Forward Pass 激活: 20GB → 24GB (+4GB)
├── offload_fsdp_model_to_cpu: 24GB → 22GB (-2GB, 如果配置了)
└── 总计: 22GB

时刻 3-6s: update_actor()
├── load_fsdp_model_to_gpu: 22GB → 24GB (+2GB)
├── load_fsdp_optimizer: 24GB → 28GB (+4GB)
├── Forward + Backward: 28GB → 38GB (+10GB, 激活 + 梯度)
├── Optimizer Step: 38GB → 38GB (in-place)
├── offload: 38GB → 22GB (-16GB)
└── 总计: 22GB

时刻 6s: 迭代结束，回到初始状态
└── 总计: 22GB

┌────────────────────────────────────────────────────────────────────┐
│                        峰值显存分析                                │
└────────────────────────────────────────────────────────────────────┘

全局峰值: 56GB (Rollout Mode + KV Cache)
  ├── rollout_mode() 中的 params.full_tensor(): 40GB
  ├── rollout_mode() 中的 resume KV Cache: 56GB ← 全局最大
  └── update_actor() 中的 Forward + Backward: 38GB

优化方向:
1. 启用 Parameter Offload: 减少 Actor 参数显存（-2GB）
2. 启用 Optimizer Offload: 减少优化器显存（-4GB）
3. 减少 KV Cache: 调整 gpu_memory_utilization（-10GB）
4. 使用 Layered Summon: 减少参数同步峰值（-10GB）
5. 减少 batch size: 减少激活值显存（-4GB）
```

---

## 8. 代码位置索引

### 8.1 资源初始化

| 功能 | 文件路径 | 行号 |
|------|---------|------|
| 创建资源池 | `verl/trainer/main_ppo.py` | 166-189 |
| 初始化 workers | `verl/trainer/ppo/ray_trainer.py` | 661-796 |
| 创建融合 Worker 类 | `verl/single_controller/ray/base.py` | 749-790 |
| 创建 RayWorkerGroup | `verl/single_controller/ray/base.py` | 361-444 |
| Spawn 机制 | `verl/single_controller/ray/base.py` | 478-512 |

### 8.2 模型初始化

| 功能 | 文件路径 | 行号 |
|------|---------|------|
| **`__init__()` Worker 初始化** | `verl/workers/fsdp_workers.py` | **139-263** |
| `init_model()` 入口 | `verl/workers/fsdp_workers.py` | 760-820 |
| **`_build_model_optimizer()` FSDP 构建** | `verl/workers/fsdp_workers.py` | **268-589** |
| - 加载 Tokenizer | `verl/workers/fsdp_workers.py` | 304-311 |
| - 确定模型 dtype | `verl/workers/fsdp_workers.py` | 314-320 |
| - 加载模型配置 | `verl/workers/fsdp_workers.py` | 323-346 |
| - 初始化模型（from_pretrained） | `verl/workers/fsdp_workers.py` | 348-389 |
| - 应用优化（Liger/Gradient Checkpointing/LoRA） | `verl/workers/fsdp_workers.py` | 392-432 |
| **- FSDP1 初始化（关键）** | `verl/workers/fsdp_workers.py` | **497-507** |
| **- FSDP2 初始化（关键）** | `verl/workers/fsdp_workers.py` | **530-532** |
| - 创建优化器 | `verl/workers/fsdp_workers.py` | 542-589 |
| **`_build_rollout()` Rollout 构建入口** | `verl/workers/fsdp_workers.py` | **591-630** |
| **- `get_rollout_class()` 动态选择引擎** | `verl/workers/rollout/base.py` | **88-93** |
| **- vLLMRollout 初始化** | `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | **92-237** |
| &nbsp;&nbsp;&nbsp;• 解析模型配置 | `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | 106-114 |
| &nbsp;&nbsp;&nbsp;• 验证 max_model_len | `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | 116-161 |
| &nbsp;&nbsp;&nbsp;• 配置 vLLM Engine 参数 | `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | 163-186 |
| &nbsp;&nbsp;&nbsp;**• 初始化 vLLM 推理引擎** | `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | **188-210** |
| &nbsp;&nbsp;&nbsp;• 配置 SamplingParams | `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | 212-227 |
| **- SGLangRollout 初始化** | `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | **249-467** |
| &nbsp;&nbsp;&nbsp;• 解析模型配置 | `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | 258-273 |
| &nbsp;&nbsp;&nbsp;• 初始化 Tools 和 Interactions | `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | 267-274 |
| &nbsp;&nbsp;&nbsp;• 初始化分布式环境 | `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | 303-346 |
| &nbsp;&nbsp;&nbsp;• 验证配置 | `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | 348-391 |
| &nbsp;&nbsp;&nbsp;**• 初始化 SGLang 推理引擎** | `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | **392-467** |
| &nbsp;&nbsp;&nbsp;• 配置 SamplingParams | `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | 482-495 |
| 配置 state_dict 类型 | `verl/workers/fsdp_workers.py` | 634-644 |
| **初始化后切换到 Trainer Mode** | `verl/workers/fsdp_workers.py` | **650-656** |

### 8.3 训练循环

| 功能 | 文件路径 | 行号 |
|------|---------|------|
| PPO 训练循环 | `verl/trainer/ppo/ray_trainer.py` | 962-1259 |
| 生成序列（调用） | `verl/trainer/ppo/ray_trainer.py` | 1042-1050 |
| 计算 Reward | `verl/trainer/ppo/ray_trainer.py` | 1088-1098 |
| 计算 Old Log Prob（调用） | `verl/trainer/ppo/ray_trainer.py` | 1100-1110 |
| 计算 Advantage | `verl/trainer/ppo/ray_trainer.py` | 1132-1164 |
| 更新 Actor（调用） | `verl/trainer/ppo/ray_trainer.py` | 1174-1180 |

### 8.4 模式切换（核心）

| 功能 | 文件路径 | 行号 |
|------|---------|------|
| **`rollout_mode()` 定义** | `verl/workers/fsdp_workers.py` | **658-739** |
| **`trainer_mode()` 定义** | `verl/workers/fsdp_workers.py` | **741-757** |
| `generate_sequences()` - 切换到 Rollout | `verl/workers/fsdp_workers.py` | **945-950** |
| `generate_sequences()` - 切换回 Trainer | `verl/workers/fsdp_workers.py` | **960-964** |

### 8.5 参数同步

| 功能 | 文件路径 | 行号 |
|------|---------|------|
| **FSDP 分片状态（ZeRO-3）** | `verl/workers/fsdp_workers.py` | **497-507** (FSDP1), **530-532** (FSDP2) |
| 配置 SHARDED_STATE_DICT | `verl/workers/fsdp_workers.py` | 634-644 |
| 收集 FSDP 参数 | `verl/workers/fsdp_workers.py` | 679 |
| **转换为 full_tensor（All-Gather）** | `verl/workers/fsdp_workers.py` | **708-714** |
| **更新 Rollout 参数** | `verl/workers/fsdp_workers.py` | **728** |
| **vLLM update_weights() 实现** | `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | **430-453** |
| **SGLang update_weights() 实现** | `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | **1505-1528** |
| 分层同步（Layered Summon） | `verl/utils/fsdp_utils.py` | 569-608 |
| 收集 LoRA 参数 | `verl/utils/fsdp_utils.py` | 611-650 |

### 8.6 Worker 方法实现

| 功能 | 文件路径 | 行号 |
|------|---------|------|
| `generate_sequences()` | `verl/workers/fsdp_workers.py` | 927-984 |
| `compute_log_prob()` | `verl/workers/fsdp_workers.py` | 986-1026 |
| `update_actor()` | `verl/workers/fsdp_workers.py` | 877-923 |

---

## 9. 性能优化建议

### 9.1 显存优化

| 优化方法 | 节省显存 | 代价 | 配置参数 |
|---------|---------|------|---------|
| Parameter Offload | 2GB | CPU↔GPU 传输延迟 ~50ms | `actor.model.enable_parameter_offload=True` |
| Optimizer Offload | 4GB | CPU↔GPU 传输延迟 ~100ms | `actor.optimizer.enable_optimizer_offload=True` |
| 减少 gpu_memory_utilization | 10-20GB | KV Cache 容量减小，batch size 受限 | `rollout.gpu_memory_utilization=0.4` |
| Layered Summon (LoRA) | 10-15GB | 增加通信次数（32x） | `rollout.layered_summon=True` |
| free_cache_engine=True | 32GB | 每次重新分配 KV Cache 开销 | `rollout.free_cache_engine=True` |
| 减少 rollout batch size | 5-10GB | 推理吞吐量降低 | `rollout.rollout_batch_size=128` |

### 9.2 通信优化

| 优化方法 | 节省时间 | 适用场景 |
|---------|---------|---------|
| 使用 NVLink/NVSwitch | ~50% | 多 GPU 同机训练 |
| 启用 NCCL 压缩 | ~20% | 低带宽网络 |
| 减少参数同步频率 | ~30% | 小步长更新 |
| Gradient Checkpointing | ~10% | 减少激活值显存，间接减少通信 |

### 9.3 推理优化

| 优化方法 | 提升吞吐 | 配置参数 |
|---------|---------|---------|
| 增加 KV Cache | ~50% | `rollout.gpu_memory_utilization=0.8` |
| 使用 PagedAttention | ~2x | vLLM/SGLang 默认启用 |
| 启用 Continuous Batching | ~3x | vLLM/SGLang 默认启用 |
| 使用 FP8 量化 | ~2x | `rollout.quantization=fp8` |

### 9.4 训练优化

| 优化方法 | 提升速度 | 配置参数 |
|---------|---------|---------|
| Gradient Accumulation | ~30% | `actor.gradient_accumulation_steps=4` |
| Mixed Precision (BF16) | ~50% | `actor.mixed_precision=bf16` |
| Flash Attention | ~2x | `actor.use_flash_attention=True` |
| Activation Checkpointing | ~20% | `actor.enable_gradient_checkpointing=True` |

---

## 10. 常见问题 (FAQ)

### Q1: 为什么初始化后要切换到 Trainer Mode？

**A**: 根据代码注释（`verl/workers/fsdp_workers.py:651`）：
> It's critical that hybrid engine in trainer mode initially to load checkpoint.

原因：
1. **加载 checkpoint** 时需要 Actor FSDP 在 GPU（用于加载参数）
2. Rollout KV Cache 在初始化时会占用大量显存（32GB），需要释放
3. 训练开始前可能需要执行验证（validation），需要 Trainer Mode

### Q2: 为什么每次 Rollout 都要同步参数，不能只同步一次？

**A**: 因为 Actor 模型在每次训练迭代后都会更新参数（`update_actor()`）。如果不同步，Rollout 会一直使用初始参数 W0，而不是最新参数 W_t，导致 PPO 的 importance sampling 失效，训练发散。

### Q3: FSDP 的 state_dict() 为什么配置为 SHARDED_STATE_DICT？

**A**:
1. **节省显存**：不在 `state_dict()` 阶段触发 all-gather
2. **延迟聚合**：在后续的 `full_tensor()` 调用时才触发 all-gather
3. **灵活性**：可以选择性地聚合某些层（layered_summon）

如果配置为 `FULL_STATE_DICT`，会在 `state_dict()` 时就触发 all-gather，峰值显存更高。

### Q4: rollout.update_weights() 为什么是 in-place 更新？

**A**:
1. **节省显存**：不需要分配新的 16GB 空间
2. **高效**：直接在 GPU 上复制（cudaMemcpy），无需 CPU 中转
3. **保持 Rollout Engine 状态**：KV Cache、PageTable 等不受影响

### Q5: 如何减少 rollout_mode() 的峰值显存？

**A**:
1. 启用 Parameter Offload：Actor FSDP offload 到 CPU（-2GB）
2. 使用 Layered Summon：逐层聚合参数（峰值从 40GB 降到 ~2GB）
3. 减少 KV Cache：调整 `gpu_memory_utilization`（-10-20GB）

### Q6: Spawn 机制的本质是什么？

**A**: Spawn 创建了多个"视图"（RayWorkerGroup 对象），它们共享相同的底层 Ray Actors（物理进程），但只暴露特定前缀的方法，实现了：
1. **资源复用**：8 个 GPU 而不是 16 个 GPU
2. **接口隔离**：逻辑上分离 Actor 和 Rollout 功能
3. **零额外开销**：只是 Python 对象层面的引用

### Q7: init 时显存中是否同时存在两份模型权重？

**A**: **是的**！在 FSDP 初始化过程中，显存中短暂存在两份权重：

**FSDP1 初始化**（`verl/workers/fsdp_workers.py:497-507`）：
```
1. from_pretrained() → 加载完整模型（32GB fp32）
2. FSDP() 调用 → 创建分片模型（2GB bf16）
   ├── 此时峰值：32GB + 2GB = 34GB
3. FSDP 内部释放完整模型 → 仅保留分片（2GB）
```

**FSDP2 初始化**（`verl/workers/fsdp_workers.py:530-532`）：
```
1. from_pretrained() → 加载完整模型（32GB fp32）
2. state_dict() → 复制完整参数（32GB）
   ├── 此时峰值：32GB + 32GB = 64GB
3. apply_fsdp2() → 应用分片
4. 释放原模型和 state_dict → 仅保留分片（2GB）
```

**总结**：
- 峰值显存：34GB (FSDP1) 或 64GB (FSDP2)
- 稳定显存：2GB (分片) + 4GB (优化器) = 6GB
- 这是 FSDP 初始化的必要开销，**只在初始化时发生一次**

### Q8: FSDP 初始化在哪里？后续每次 rollout 完都会重新初始化 FSDP 吗？

**A**: **不会**！FSDP 只在 `init_model()` 时初始化一次。

**FSDP 初始化位置**：
- `verl/workers/fsdp_workers.py:497-507` (FSDP1)
- `verl/workers/fsdp_workers.py:530-532` (FSDP2)
- 调用时机：Ray Trainer 启动时调用 `init_model()`（**只执行一次**）

**后续训练流程**：
```
初始化阶段（只执行一次）：
├── __init__() → 初始化分布式环境
├── init_model() → 加载模型
│   ├── _build_model_optimizer() → FSDP 初始化 ← 只执行一次
│   └── _build_rollout() → Rollout 初始化 ← 只执行一次
└── trainer_mode() → 切换到 Trainer Mode

训练迭代（重复执行）：
├── generate_sequences()
│   ├── rollout_mode() → 同步参数到 Rollout（不重新初始化 FSDP）
│   ├── Rollout 推理
│   └── trainer_mode() → 切换回 Trainer Mode
├── compute_log_prob() → 使用已有的 FSDP 模型
├── update_actor() → 更新已有的 FSDP 参数
└── 下一轮迭代...
```

**关键点**：
1. **FSDP 只初始化一次**（在 `init_model()` 中）
2. **参数同步不是重新初始化**：
   - `rollout_mode()` 只是将 FSDP 的**当前参数**复制到 Rollout
   - 不会重新调用 `FSDP()` 或 `apply_fsdp2()`
3. **模式切换不改变 FSDP 结构**：
   - `rollout_mode()` 和 `trainer_mode()` 只控制显存分配（KV Cache）
   - FSDP 模型结构保持不变

### Q9: Actor FSDP 模型和 Rollout 模型的参数是独立的吗？

**A**: **是的**，它们是两份独立的权重：

**物理存储**：
```
GPU 0:
├── Actor FSDP 参数分片: 2GB (bf16)
│   └── 存储：GPU 显存
│   └── 用途：训练（梯度计算、参数更新）
│
├── Rollout 参数（完整）: 16GB (bf16)
    └── 存储：GPU 显存
    └── 用途：推理（生成序列）
```

**为什么需要两份权重**：
1. **FSDP 模型是分片的**：
   - 每个 GPU 只有 1/8 的参数（2GB）
   - 无法直接用于推理（缺少其他 GPU 的参数）
   - 训练时通过 All-Gather 临时聚合完整参数

2. **Rollout 模型是完整的**：
   - 每个 GPU 有完整的 8B 参数（16GB）
   - 推理时可以独立工作，无需跨 GPU 通信
   - vLLM/SGLang 需要完整模型才能高效推理

**参数同步流程**：
```
训练更新 → Actor FSDP 参数改变（2GB 分片）
   ↓
rollout_mode() 调用
   ↓
1. full_tensor() → All-Gather 聚合所有 GPU 的分片（临时 16GB）
2. update_weights() → 复制到 Rollout 模型（16GB）
3. 释放临时聚合的参数
   ↓
Rollout 模型参数已更新（16GB）
```

**总结**：
- 两份权重独立存储（2GB FSDP 分片 + 16GB Rollout 完整）
- 通过参数同步保持一致（每次 rollout 前同步）
- 总显存：2GB + 16GB = 18GB（不包括优化器和 KV Cache）

### Q10: FSDP 分片后每个 GPU 的 actor_module_fsdp 是不是不完整的模型？

**A**: **是的！** 在 FSDP 初始化后（ZeRO-3 模式），每个 GPU 上的 `actor_module_fsdp` 只包含部分参数（分片），不是完整模型。

**详细解释**（见 6.2 节）：

```python
# 模型: Qwen3-8B (8B 参数, bf16 = 16GB)
# GPU 数量: 8 个 GPU
# FSDP 策略: FULL_SHARD (ZeRO-3)

# 每个 GPU 上的参数分片:
GPU 0: 分片 0 (2GB, 参数 0-1B)
GPU 1: 分片 1 (2GB, 参数 1-2B)
GPU 2: 分片 2 (2GB, 参数 2-3B)
...
GPU 7: 分片 7 (2GB, 参数 7-8B)

# 对于单个 GPU 进程而言:
# - actor_module_fsdp 只包含本地分片（2GB）
# - 无法直接用于推理（参数不完整）
# - 训练时通过 All-Gather 临时聚合完整参数
```

**参数聚合机制**（见 6.4 节）：
- 调用 `param.full_tensor()` 时触发 NCCL All-Gather
- 聚合所有 GPU 的分片，得到完整参数（16GB）
- 复制到 Rollout 模型（in-place 更新）

### Q11: vLLM 和 SGLang 的 update_weights() 实现一样吗？

**A**: **基本一样，但有重要区别！**

**共同点**（见 6.5.3 节）：
1. 都使用 `param.data.copy_()` in-place 更新（不需要额外显存）
2. 都保持 Rollout Engine 的其他状态（KV Cache 不受影响）
3. 总耗时接近：vLLM ~100ms，SGLang ~120ms

**关键区别**（见 6.5.1 和 6.5.2 节）：

| 特性 | vLLM | SGLang |
|------|------|--------|
| **更新策略** | 一次性传输所有参数 | 分批传输（buckets） |
| **单次调用** | 1 次 `model.load_weights()` | N 次 `sgl_update_weights()` |
| **Bucket 大小** | 完整 16GB | 可配置（默认 256MB） |
| **Ray object store 压力** | 高（16GB 单个对象） | 低（256MB 小对象） |
| **适用场景** | 中等模型（7B-70B） | 超大模型（100B+） |

**实现代码对比**：

```python
# vLLM (verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:430-453):
async def update_weights(self, weights, **kwargs):
    model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
    model.load_weights(weights)  # 一次性传输所有参数

# SGLang (verl/workers/rollout/sglang_rollout/sglang_rollout.py:1505-1528):
async def update_weights(self, weights, **kwargs):
    bucket_bytes = int(self.config.update_weights_bucket_megabytes) << 20
    for params_batch in get_named_tensor_buckets(weights, bucket_bytes):
        await sgl_update_weights(engine=self._engine, params_batch=params_batch)
        # 分批传输，64 次迭代（16GB / 256MB）
```

**选择建议**：
- **7B-70B 模型**：使用 vLLM（更简单，性能更好）
- **100B+ 模型**：使用 SGLang（分批更新更稳定）
- **Ray 内存受限**：使用 SGLang（避免大对象传输）

### Q12: 为什么 trainer_mode() 只需要 self.actor_module_fsdp.train() 这么简单？

**A**: 因为所有训练需要的权重和状态都已经在内存中了！（见 6.6 节）

**关键理解**：

1. **参数不需要重新加载**：
   - Actor FSDP 的参数分片（2GB）始终在 GPU（或 CPU 如果 offload）
   - `update_actor()` 更新的就是这些参数
   - 下一次 `rollout_mode()` 会同步这些更新后的参数到 Rollout

2. **优化器状态不需要重新加载**：
   - AdamW 的 momentum 和 variance（4GB）一直在内存
   - 每次 `update_actor()` 会更新优化器状态

3. **train() 只是模式切换**：
   ```python
   # PyTorch nn.Module.train() 的作用：
   # 1. 设置 self.training = True
   # 2. 递归调用所有子模块的 train()
   # 3. 启用训练特性：
   #    - Dropout: 在训练时随机丢弃神经元（inference 时不丢弃）
   #    - BatchNorm: 更新 running_mean 和 running_var（inference 时使用固定值）
   #    - LayerNorm: 无影响（train 和 eval 行为一致）
   #
   # 重要：train() 不涉及参数的加载或卸载！
   ```

4. **为什么 Rollout 参数还在 GPU？**：
   - Rollout 参数（16GB）保留在 GPU，但在 Trainer Mode 不使用
   - 这是为了避免频繁的参数加载/卸载（开销大）
   - 只释放 KV Cache（32GB），因为它占用最多且训练时不需要

**完整的资源状态对比**（见 6.6 节）：

```
Trainer Mode (update_actor, compute_log_prob):
├── Actor FSDP 分片: 2GB (使用中，在 GPU)
├── Actor 优化器: 4GB (使用中，在 GPU)
├── Rollout 参数: 16GB (不使用，在 GPU)  ← 保留但不使用
├── Rollout KV Cache: 0GB (已释放)       ← trainer_mode() 释放
├── self.actor_module_fsdp.training = True
└── 总计: 22GB

Rollout Mode (generate_sequences):
├── Actor FSDP 分片: 2GB (不使用，可 offload 到 CPU)
├── Actor 优化器: 4GB (不使用，可 offload 到 CPU)
├── Rollout 参数: 16GB (使用中，已更新为最新)
├── Rollout KV Cache: 32GB (已分配，使用中)
├── self.actor_module_fsdp.training = False
└── 总计: 48-54GB（取决于 offload）
```

**总结**：
`trainer_mode()` 只需要 `train()` 这么简单，是因为：
1. 所有训练权重（FSDP 分片 + 优化器）始终在内存
2. `train()` 只是切换模块的行为模式（Dropout/BatchNorm）
3. 实际的资源释放只发生在 KV Cache（`rollout.release()`）
4. Rollout 参数保留在 GPU，避免频繁加载/卸载的开销

---

## 11. 总结

### 核心要点

1. **Hybrid Engine 架构**：
   - 在同一组 GPU 上共置 Actor FSDP 和 Rollout 模型
   - 通过动态切换 Trainer Mode 和 Rollout Mode 复用显存
   - 节省 50% GPU 资源

2. **模式切换机制**：
   - **初始化后**：系统处于 **Trainer Mode**（`verl/workers/fsdp_workers.py:650-656`）
   - **每次 generate_sequences()**：
     - 开始时调用 `rollout_mode()`（`verl/workers/fsdp_workers.py:945-950`）
     - 结束时调用 `trainer_mode()`（`verl/workers/fsdp_workers.py:960-964`）
   - **compute_log_prob() 和 update_actor()**：无需切换（已在 Trainer Mode）

3. **参数同步流程**：
   - Actor FSDP `state_dict()` → 返回分片参数（DTensor）
   - `full_tensor()` → All-Gather 聚合为完整参数（NCCL 通信 ~200ms）
   - `rollout.update_weights()` → In-Place 复制到 Rollout 模型（~100ms）
   - **必须每次同步**：因为 Actor 参数在每次训练后都更新

4. **显存峰值管理**：
   - 全局峰值：56GB（Rollout Mode + KV Cache）
   - 优化方向：Parameter/Optimizer Offload、Layered Summon、减少 KV Cache

5. **Spawn 共置机制**：
   - 物理层面：8 个 Ray Actors（共享）
   - 逻辑层面：多个 RayWorkerGroup 视图（隔离）
   - 目的：避免为每个功能创建独立 GPU 进程

---

**文档版本**: v1.0
**最后更新**: 2025-01-XX
**适用 verl 版本**: v0.1.0+
