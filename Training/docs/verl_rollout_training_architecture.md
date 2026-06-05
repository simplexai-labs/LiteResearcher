# verl Rollout 与训练架构详解

本文档基于 `examples/sglang_multiturn/search_browser/qwen3_agentloop.sh` 脚本，详细解析 verl 框架的 rollout 和训练实现。

---

## 目录

1. [整体架构概览](#一整体架构概览)
2. [Rollout 实现详解](#二rollout-实现详解)
3. [训练初始化详解](#三训练初始化详解) ⭐ **新增**
4. [训练主循环](#四训练主循环)
5. [训练提速技术](#五训练提速技术)
6. [针对当前配置的优化建议](#六针对当前配置的优化建议) ⭐ **新增**
7. [关键数据流](#七关键数据流)
8. [配置总结](#八配置总结)
9. [总结](#九总结)

---

## 一、整体架构概览

verl 是一个分布式强化学习训练框架，基于 Ray 实现多节点多 GPU 的分布式训练。其核心架构包括：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RayPPOTrainer (主控制器)                            │
│  - 协调所有训练流程                                                           │
│  - 管理数据加载、奖励计算、优势估计                                            │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ ActorRollout  │   │   Critic      │   │  RefPolicy    │
│   Worker      │   │   Worker      │   │   Worker      │
│ (FSDP/Megatron)│   │ (FSDP/Megatron)│   │ (可选)        │
└───────┬───────┘   └───────────────┘   └───────────────┘
        │
        ▼ (async mode)
┌───────────────────────────────────────┐
│      AgentLoopManager                 │
│  - 管理多个 AgentLoopWorker           │
│  - 调度 SGLang HTTP Server            │
└───────────────────────────────────────┘
```

---

## 二、Rollout 实现详解

### 2.1 Rollout 本质

**回答你的问题：Rollout 本质就是调用 SGLang Router 来写 AgentLoop 逻辑吗？**

**是的，但不完全是。** Rollout 的实现分为两种模式：

1. **同步模式 (sync mode)**: 直接调用 SGLang Engine 进行批量生成
2. **异步模式 (async mode)**: 通过 AgentLoopManager 调度多个 AgentLoopWorker，每个 Worker 通过 HTTP 与 SGLang Server 交互

从你的脚本配置可以看到使用的是异步模式：
```bash
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
```

### 2.2 异步 Rollout 架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    AgentLoopManager                              │
│  文件: verl/experimental/agent_loop/agent_loop.py               │
│  功能:                                                          │
│  - 初始化 SGLang HTTP Server 集群                               │
│  - 创建多个 AgentLoopWorker                                     │
│  - 将 batch 拆分并分发到各个 Worker                             │
└─────────────────────────┬────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│AgentLoopWorker│ │AgentLoopWorker│ │AgentLoopWorker│
│   (Ray Actor) │ │   (Ray Actor) │ │   (Ray Actor) │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│               SGLang HTTP Server 集群                            │
│  - 每个节点运行一个 SGLangHttpServer                             │
│  - 提供 /generate 接口进行文本生成                               │
│  - 支持 KV Cache 管理 (wake_up/sleep)                           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 ToolAgentLoop - 多轮对话的核心

文件位置: `verl/experimental/agent_loop/tool_agent_loop.py`

这是实现多轮工具调用的核心类，采用**状态机模式**：

```python
class AgentState(Enum):
    PENDING = "pending"           # 等待开始
    GENERATING = "generating"     # 模型生成中
    PROCESSING_TOOLS = "processing_tools"  # 处理工具调用
    TERMINATED = "terminated"     # 终止
    INTERACTING = "interacting"   # 交互中(可选)
```

**状态机流程:**

```
PENDING → GENERATING → PROCESSING_TOOLS → GENERATING → ... → TERMINATED
                ↓                                ↓
          (检测到<answer>)              (达到max_turns)
                ↓                                ↓
           TERMINATED                       TERMINATED
```

**核心流程代码:**

```python
async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
    # 状态机循环
    state = AgentState.PENDING
    while state != AgentState.TERMINATED:
        if state == AgentState.PENDING:
            state = await self._handle_pending_state(agent_data, sampling_params)
        elif state == AgentState.GENERATING:
            state = await self._handle_generating_state(agent_data, sampling_params)
        elif state == AgentState.PROCESSING_TOOLS:
            state = await self._handle_processing_tools_state(agent_data)
        # ...
```

### 2.4 Rollout 配置参数详解

从脚本中可以看到关键配置：

```bash
# SGLang 配置
actor_rollout_ref.rollout.name=sglang                    # 使用 SGLang 作为推理引擎
actor_rollout_ref.rollout.mode=async                     # 异步模式
actor_rollout_ref.rollout.max_model_len=49152            # 最大序列长度
actor_rollout_ref.rollout.gpu_memory_utilization=0.92    # GPU 显存利用率
actor_rollout_ref.rollout.tensor_model_parallel_size=1   # TP 并行度

# 多轮对话配置
actor_rollout_ref.rollout.multi_turn.enable=true
actor_rollout_ref.rollout.multi_turn.terminate_on_answer=True    # 检测到<answer>停止
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20      # 最大轮数
actor_rollout_ref.rollout.multi_turn.max_tool_response_length=100000  # 工具响应最大长度
actor_rollout_ref.rollout.multi_turn.format=hermes               # 对话格式
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent    # 使用工具Agent
```

---

## 三、训练初始化详解

训练初始化是一个多层级的过程，从 `main_ppo.py` 入口开始，逐层创建 Ray Actors 和初始化模型。

### 3.1 完整初始化流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        python -m verl.trainer.main_ppo                       │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           1. Ray 初始化                                      │
│  ray.init() - 启动 Ray 集群                                                  │
│  设置环境变量: TOKENIZERS_PARALLELISM, VLLM_LOGGING_LEVEL 等                 │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        2. TaskRunner.run()                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  2.1 add_actor_rollout_worker()                                     │    │
│  │      根据 actor.strategy (fsdp/megatron) 选择 Worker 类             │    │
│  │      async mode → AsyncActorRolloutRefWorker                        │    │
│  │      sync mode  → ActorRolloutRefWorker                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  2.2 add_critic_worker() - GRPO 算法不需要                          │    │
│  │  2.3 add_reward_model_worker() - 使用自定义reward_fn时不需要        │    │
│  │  2.4 add_ref_policy_worker() - 当 use_kl_loss=True 时需要           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  2.5 init_resource_pool_mgr()                                       │    │
│  │      创建 GPU 资源池: global_pool = [8] * 1 (8卡×1节点)             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  2.6 创建 RayPPOTrainer                                             │    │
│  │      - 加载 tokenizer/processor                                     │    │
│  │      - 创建 reward_fn (LLM Judge)                                   │    │
│  │      - 创建 train_dataloader / val_dataloader                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     3. trainer.init_workers()                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  3.1 创建 RayResourcePool                                           │    │
│  │      - 分配 GPU 到各个角色 (ActorRollout, Critic, RefPolicy)        │    │
│  │      - 验证 GPU 资源是否足够                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  3.2 创建 Colocated Worker 类                                       │    │
│  │      - create_colocated_worker_cls() 合并多个角色到同一进程         │    │
│  │      - 节省 GPU 显存，允许训练和推理共享权重                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  3.3 spawn RayWorkerGroup                                           │    │
│  │      - 为每个 GPU 创建一个 Ray Actor                                │    │
│  │      - 8 GPUs → 8 个 ActorRolloutRefWorker 实例                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  3.4 各 Worker 初始化模型                                           │    │
│  │      actor_rollout_wg.init_model()                                  │    │
│  │      critic_wg.init_model() (if use_critic)                         │    │
│  │      ref_policy_wg.init_model() (if use_kl_loss)                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  3.5 创建 AgentLoopManager (async mode only)                        │    │
│  │      - 启动 SGLang HTTP Server                                      │    │
│  │      - 创建 AgentLoopWorker 池                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        4. trainer.fit()                                      │
│                        开始训练循环                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 ActorRolloutRefWorker.init_model() 详解

这是最关键的初始化步骤，文件位置: `verl/workers/fsdp_workers.py`

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)  # 所有 GPU 并行执行
def init_model(self):
    # ==================== 阶段1: 准备配置 ====================
    override_model_config = self.config.model.get("override_config", {})
    use_remove_padding = self.config.model.get("use_remove_padding", False)  # 移除padding提速
    use_shm = self.config.model.get("use_shm", False)  # 使用共享内存加速加载
    use_fused_kernels = self.config.model.get("use_fused_kernels", False)  # 融合算子
    
    # ==================== 阶段2: 构建 Actor 模型 ====================
    if self._is_actor or self._is_rollout:
        # 2.1 从远程/本地复制模型到本地
        local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
        
        # 2.2 构建模型+优化器 (关键耗时步骤)
        (
            self.actor_module_fsdp,   # FSDP 包装后的模型
            self.actor_optimizer,     # AdamW/FusedAdam 优化器
            self.actor_lr_scheduler,  # 学习率调度器
            self.actor_model_config,  # HF 模型配置
        ) = self._build_model_optimizer(
            model_path=local_path,
            fsdp_config=fsdp_config,
            optim_config=optim_config,
            use_remove_padding=use_remove_padding,
            use_fused_kernels=use_fused_kernels,
            enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
            use_liger=self.config.model.get("use_liger", False),  # Liger Kernel
        )
        
        # 2.3 参数卸载到 CPU (如果配置了 param_offload)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            # 日志: "After offload actor model during init"
        
        # 2.4 优化器状态卸载到 CPU (如果配置了 optimizer_offload)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
    
    # ==================== 阶段3: 创建 PPO Actor ====================
    if self._is_actor:
        self.actor = DataParallelPPOActor(
            config=actor_cfg,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer
        )
    
    # ==================== 阶段4: 构建 Rollout 引擎 ====================
    if self._is_rollout:
        self._build_rollout(trust_remote_code=True)
        # 内部会初始化 SGLang Engine 或连接 HTTP Server
    
    # ==================== 阶段5: 构建 Reference Policy ====================
    if self._is_ref:
        # 加载 ref 模型 (可以和 actor 共享权重，也可以独立加载)
        self.ref_module_fsdp = self._build_model_optimizer(
            model_path=local_path,
            fsdp_config=self.config.ref.fsdp_config,
            optim_config=None,  # ref 不需要优化器
        )[0]
        self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
    
    # ==================== 阶段6: 创建辅助组件 ====================
    if self._is_actor:
        self.flops_counter = FlopsCounter(self.actor_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.actor_module_fsdp,
            optimizer=self.actor.actor_optimizer,
            lr_scheduler=self.actor_lr_scheduler,
        )
```

### 3.3 _build_model_optimizer() 内部流程

```python
def _build_model_optimizer(self, model_path, fsdp_config, optim_config, ...):
    # Step 1: 加载 HuggingFace 模型
    with get_init_weight_context_manager():  # 延迟初始化权重
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # 使用 bf16
            attn_implementation="flash_attention_2",  # Flash Attention
        )
    
    # Step 2: 应用 Liger Kernel (如果启用)
    if use_liger:
        apply_liger_kernel_to_model(model)  # 替换为高效 Triton 内核
    
    # Step 3: 启用梯度检查点 (如果启用)
    if enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Step 4: 用 FSDP 包装模型
    auto_wrap_policy = get_fsdp_wrap_policy(model)  # 自动确定分片策略
    model_fsdp = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        cpu_offload=CPUOffload(offload_params=fsdp_config.param_offload),
    )
    
    # Step 5: 创建优化器
    if optim_config:
        optimizer = torch.optim.AdamW(
            model_fsdp.parameters(),
            lr=optim_config.lr,
            betas=(0.9, 0.999),
            weight_decay=optim_config.weight_decay,
        )
        # 或使用 FusedAdam (更快)
    
    # Step 6: 创建学习率调度器
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    return model_fsdp, optimizer, lr_scheduler, model.config
```

### 3.4 AgentLoopManager 初始化 (异步模式)

```python
# 文件: verl/experimental/agent_loop/agent_loop.py
class AgentLoopManager:
    def __init__(self, config, worker_group, rm_wg=None):
        self.config = config
        self.worker_group = worker_group
        
        # Step 1: 初始化 SGLang HTTP Server 集群
        self._initialize_llm_servers()
        
        # Step 2: 创建 AgentLoopWorker 池
        self._init_agent_loop_workers()
        
        # Step 3: 初始睡眠模式 (释放 KV Cache)
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()
    
    def _initialize_llm_servers(self):
        """启动 SGLang HTTP Server"""
        # 计算 rollout 并行度
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size *
            self.config.actor_rollout_ref.rollout.data_parallel_size
        )
        
        # 创建 SGLangReplica
        for replica_rank in range(num_replicas):
            replica = SGLangReplica(
                config=self.config.actor_rollout_ref.rollout,
                model_config=model_config,
                workers=workers,
            )
            # 启动 HTTP Server
            await replica.launch_servers()
            self.replicas.append(replica)
    
    def _init_agent_loop_workers(self):
        """创建并行的 AgentLoopWorker"""
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers
        
        for i in range(num_workers):
            worker = AgentLoopWorker.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(...)
            ).remote(
                config=self.config,
                server_handles=self.server_handles,
                rm_executor=self.rm_executor,
            )
            self.agent_loop_workers.append(worker)
```

### 3.5 初始化耗时分析

基于你的配置，初始化各阶段预估耗时：

| 阶段 | 操作 | 预估耗时 | 主要瓶颈 |
|------|------|---------|---------|
| 1 | Ray 初始化 | ~10s | 网络通信 |
| 2 | 模型加载 (HF) | ~30-60s | 磁盘I/O, 模型大小 |
| 3 | FSDP 包装 | ~20-30s | 参数分片, 同步 |
| 4 | 参数卸载到CPU | ~10s | 内存带宽 |
| 5 | SGLang Server 启动 | ~30-60s | 模型加载, KV Cache 分配 |
| 6 | AgentLoopWorker 创建 | ~5s | Ray Actor 创建 |
| **总计** | | **~2-4 分钟** | |

## 四、训练主循环

文件位置: `verl/trainer/ppo/ray_trainer.py` - `init_workers()` 方法

```python
def init_workers(self):
    # 1. 创建 Ray 资源池
    self.resource_pool_manager.create_resource_pool()
    
    # 2. 配置 Actor/Rollout Worker
    if self.hybrid_engine:
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )
    
    # 3. 创建协同 Worker 类
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    wg_dict = self.ray_worker_group_cls(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
    )
    
    # 4. 初始化模型
    self.actor_rollout_wg.init_model()
    
    # 5. 创建异步 Rollout Manager (如果是异步模式)
    if self.config.actor_rollout_ref.rollout.mode == "async":
        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_wg=self.rm_wg
        )
```

文件位置: `verl/trainer/ppo/ray_trainer.py` - `fit()` 方法

```python
def fit(self):
    # 加载检查点 (支持断点续训)
    self._load_checkpoint()
    
    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            metrics = {}
            timing_raw = {}
            
            # ========== 1. Rollout 生成 (最耗时) ==========
            with marked_timer("gen", timing_raw):
                if self.async_rollout_mode:
                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                else:
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            
            # ========== 2. 奖励计算 ==========
            with marked_timer("reward", timing_raw):
                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(batch, self.reward_fn)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
            
            # ========== 3. 计算 old_log_prob ==========
            with marked_timer("old_log_prob", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            
            # ========== 4. 计算 ref_log_prob (用于 KL 约束) ==========
            if self.use_reference_policy:
                with marked_timer("ref", timing_raw):
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
            
            # ========== 5. 计算优势估计 ==========
            with marked_timer("adv", timing_raw):
                # 等待异步奖励计算完成
                if self.config.reward_model.launch_reward_fn_async:
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                
                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,  # GRPO
                )
            
            # ========== 6. 更新 Actor ==========
            if self.config.trainer.critic_warmup <= self.global_steps:
                with marked_timer("update_actor", timing_raw):
                    actor_output = self.actor_rollout_wg.update_actor(batch)
            
            # ========== 7. 保存检查点 ==========
            if self.global_steps % self.config.trainer.save_freq == 0:
                self._save_checkpoint()
            
            self.global_steps += 1
```

### 4.1 单步训练耗时分析

基于你的配置 (`train_batch_size=128, n=8, max_response_length=45056`)：

| 阶段 | 操作 | 预估耗时占比 | 主要瓶颈 |
|------|------|-------------|---------|
| gen | Rollout 生成 | **60-70%** | 序列长度、多轮工具调用 |
| reward | 奖励计算 | **10-15%** | LLM Judge API 调用 |
| old_log_prob | 计算 log prob | 5-10% | 显存、计算 |
| ref | Reference log prob | 3-5% | 显存、计算 |
| update_actor | 参数更新 | 10-15% | CPU卸载开销 |

### 4.2 FSDP Worker 实现

文件位置: `verl/workers/fsdp_workers.py`

**Actor/Rollout 混合 Worker 架构:**

```python
class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    这个 Worker 可以实例化为:
    - 独立的 Actor
    - 独立的 Rollout
    - 独立的 Reference Policy
    - 混合引擎 (同时包含以上功能)
    """
    
    def __init__(self, config: DictConfig, role: str, **kwargs):
        # 初始化分布式环境
        torch.distributed.init_process_group(...)
        
        # 创建 FSDP device mesh
        self.device_mesh = create_device_mesh(world_size, fsdp_size)
        
        # 标记角色
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
```

---

## 五、训练提速技术

### 5.1 显存优化

#### 4.1.1 FSDP 参数分片与卸载

```bash
# 从脚本配置
actor_rollout_ref.actor.fsdp_config.param_offload=True      # 参数卸载到 CPU
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True  # 优化器状态卸载到 CPU
actor_rollout_ref.ref.fsdp_config.param_offload=True        # Ref 模型卸载
```

**实现机制:**
```python
# 文件: verl/workers/fsdp_workers.py
async def trainer_mode(self):
    """切换到训练模式，将模型加载到 GPU"""
    if self._is_offload_param:
        load_fsdp_model_to_gpu(self.actor_module)

async def rollout_mode(self):
    """切换到 Rollout 模式，将训练模型卸载到 CPU"""
    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.actor_module)
```

#### 4.1.2 梯度检查点 (Gradient Checkpointing)

```bash
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

在前向传播时不保存中间激活值，在反向传播时重新计算，用时间换空间。

#### 4.1.3 使用 Liger Kernel

```bash
actor_rollout_ref.model.use_liger=True
actor_rollout_ref.model.use_remove_padding=True
```

Liger Kernel 提供了内存高效的 Triton 内核实现，减少显存占用。

### 5.2 计算优化

#### 4.2.1 混合精度训练

```bash
# 默认使用 bf16/fp16 混合精度
actor_rollout_ref.actor.fsdp_config.mixed_precision=True
```

#### 4.2.2 Micro Batch 累积

```bash
actor_rollout_ref.actor.ppo_mini_batch_size=64            # mini batch 大小
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1    # 每 GPU micro batch
```

当 `ppo_micro_batch_size_per_gpu < ppo_mini_batch_size / world_size` 时，会使用梯度累积。

#### 4.2.3 序列长度负载均衡

```python
# 文件: verl/trainer/ppo/ray_trainer.py
def _balance_batch(self, batch: DataProto, metrics):
    """重新排序数据，使每个 DP rank 获得相似的总 token 数"""
    global_seqlen_lst = batch.batch["attention_mask"].sum(-1).tolist()
    global_partition_lst = get_seqlen_balanced_partitions(
        global_seqlen_lst, k_partitions=world_size, equal_size=True
    )
    batch.reorder(global_idx)
```

### 5.3 并行策略

#### 4.3.1 数据并行 (FSDP)

```bash
# FSDP 自动管理参数分片和梯度同步
trainer.n_gpus_per_node=8
trainer.nnodes=1
```

#### 4.3.2 张量并行 (用于推理)

```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=1  # 推理时的 TP 并行度
```

#### 4.3.3 异步 Rollout

```bash
actor_rollout_ref.rollout.mode=async
```

异步模式下，多个 AgentLoopWorker 并行处理不同的请求，提高吞吐量。

### 5.4 KV Cache 管理

```python
# 文件: verl/experimental/agent_loop/agent_loop.py
class AgentLoopManager:
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()   # 加载 KV Cache
        
        # ... 执行生成 ...
        
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()     # 释放 KV Cache
```

### 5.5 训练-推理模式切换

在混合引擎模式下，同一组 GPU 既用于训练也用于推理：

```python
# 文件: verl/workers/fsdp_workers.py
def generate_sequences(self, prompts: DataProto):
    if self._is_actor:  # 混合模式
        # 切换到 rollout 模式，卸载训练参数
        loop.run_until_complete(self.rollout_mode())
    
    # 执行推理
    output = self.rollout.generate_sequences(prompts=prompts)
    
    if self._is_actor:
        # 切换回训练模式，加载训练参数
        loop.run_until_complete(self.trainer_mode())
```

---

## 六、针对当前配置的优化建议 ⭐

根据你的脚本配置分析，以下是导致训练慢的主要原因和优化建议：

### 6.1 当前配置瓶颈分析

```bash
# 你的当前配置
data.train_batch_size=128
data.max_response_length=45056          # ⚠️ 超长序列！45K tokens
actor_rollout_ref.rollout.n=8           # 每个prompt采样8次
actor_rollout_ref.rollout.max_model_len=49152  # ⚠️ 49K 上下文
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1  # ⚠️ 显存紧张信号
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20  # 最多20轮
actor_rollout_ref.rollout.multi_turn.max_tool_response_length=100000  # 10万字符
```

**主要瓶颈：**

| 瓶颈 | 原因 | 影响 |
|------|------|------|
| 🔴 超长序列 | `max_response_length=45056` | Rollout 生成极慢，显存占用高 |
| 🔴 CPU 卸载 | `param_offload=True` | 每次更新需要 CPU↔GPU 数据传输 |
| 🟡 多轮工具调用 | `max_assistant_turns=20` | 需要多次网络请求 |
| 🟡 LLM Judge | `custom_reward_function` | API 调用延迟 |
| 🟢 采样数 | `n=8` | 合理，但增加计算量 |

### 6.2 高优先级优化建议

#### 6.2.1 减少序列长度 (最重要！)

```bash
# 当前配置
data.max_response_length=45056  # 45K tokens

# 建议优化
data.max_response_length=16384  # 或 8192，根据实际需求调整
actor_rollout_ref.rollout.max_model_len=20480  # 相应减小

# 收益: Rollout 速度提升 2-4x，显存压力大幅降低
```

#### 6.2.2 增加 micro_batch_size (如果显存允许)

```bash
# 当前配置
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1

# 尝试增加
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2  # 或 4

# 收益: 减少梯度累积次数，GPU 利用率提升
```

#### 6.2.3 关闭 CPU 卸载 (如果显存足够)

```bash
# 当前配置
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

# 如果显存足够，尝试关闭
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

# 收益: 消除 CPU↔GPU 数据传输开销，训练速度提升 30-50%
```

#### 6.2.4 减少多轮次数

```bash
# 当前配置
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20

# 建议优化
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10  # 或 8

# 收益: 减少工具调用次数和生成轮数
```

#### 6.2.5 启用异步奖励计算

```bash
# 添加配置
reward_model.launch_reward_fn_async=True

# 收益: LLM Judge 与其他计算并行执行
```

### 6.3 中等优先级优化

#### 6.3.1 增加 log_prob_micro_batch_size

```bash
# 当前配置
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4

# 尝试增加
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8  # 或 16

# 收益: compute_log_prob 阶段提速
```

#### 6.3.2 使用 Flash Attention 3

确保 SGLang 使用 FA3：
```bash
# 默认已启用，检查日志确认
# attention_backend: fa3
```

#### 6.3.3 截断工具响应

```bash
# 当前配置
actor_rollout_ref.rollout.multi_turn.max_tool_response_length=100000

# 建议减小
actor_rollout_ref.rollout.multi_turn.max_tool_response_length=30000  # 或更小

# 收益: 减少每轮的 token 数量
```

### 6.4 低优先级优化

#### 6.4.1 使用 Tensor Parallel (如果模型更大)

```bash
# 当前配置
actor_rollout_ref.rollout.tensor_model_parallel_size=1

# 对于大模型，可以增加
actor_rollout_ref.rollout.tensor_model_parallel_size=2  # 或 4
```

#### 6.4.2 调整 GPU 显存利用率

```bash
# 当前配置
actor_rollout_ref.rollout.gpu_memory_utilization=0.92

# 可以尝试
actor_rollout_ref.rollout.gpu_memory_utilization=0.95  # 激进一点
```

### 6.5 优化配置示例

基于上述分析，推荐的优化配置：

```bash
python3 -m verl.trainer.main_ppo \
    # ... 其他配置保持不变 ...
    
    # 关键优化
    data.max_response_length=16384 \  # 减小序列长度
    actor_rollout_ref.rollout.max_model_len=20480 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \  # 尝试增加
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \  # 减少轮数
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=30000 \
    reward_model.launch_reward_fn_async=True \  # 异步奖励
    
    # 如果显存足够，关闭卸载
    # actor_rollout_ref.actor.fsdp_config.param_offload=False \
    # actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
```

### 6.6 性能监控

训练过程中关注以下指标（查看 wandb 日志）：

```
# 时间指标
timing/gen           # Rollout 生成时间 - 应该是主要瓶颈
timing/reward        # 奖励计算时间
timing/old_log_prob  # Log prob 计算时间
timing/update_actor  # 参数更新时间

# 显存指标
perf/max_memory_allocated_gb  # 峰值显存
perf/max_memory_reserved_gb   # 预留显存

# 效率指标
perf/mfu/actor  # 模型 FLOPS 利用率
```

---

## 七、关键数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           训练数据流                                         │
└─────────────────────────────────────────────────────────────────────────────┘

1. 数据加载
   train_dataloader → batch_dict → DataProto

2. Rollout 生成
   DataProto → AgentLoopManager → [AgentLoopWorker × N] → SGLang Server
            → gen_batch_output (包含 prompt_ids, response_ids, response_mask)

3. 奖励计算
   batch + gen_batch_output → reward_fn (LLM Judge) → token_level_scores

4. Log Prob 计算
   batch → ActorRolloutWorker.compute_log_prob() → old_log_probs
   batch → ActorRolloutWorker.compute_ref_log_prob() → ref_log_prob

5. 优势估计 (GRPO)
   token_level_rewards + response_mask + uid → compute_grpo_outcome_advantage
                                            → advantages, returns

6. Actor 更新
   batch (with advantages) → ActorRolloutWorker.update_actor()
                          → 梯度计算 → 参数更新

7. 保存检查点
   actor_module → FSDPCheckpointManager.save_checkpoint()
```

---

## 八、配置总结

### 8.1 核心脚本参数

| 参数 | 作用 | 值 |
|------|------|-----|
| `algorithm.adv_estimator` | 优势估计算法 | grpo |
| `data.train_batch_size` | 训练批次大小 | 128 |
| `actor_rollout_ref.rollout.n` | 每个 prompt 采样数 | 8 |
| `actor_rollout_ref.rollout.mode` | Rollout 模式 | async |
| `actor_rollout_ref.actor.use_kl_loss` | 是否使用 KL 损失 | True |
| `actor_rollout_ref.actor.kl_loss_coef` | KL 损失系数 | 0.001 |

### 8.2 提速相关参数

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `param_offload` | CPU 卸载参数 | True (显存不足时) |
| `optimizer_offload` | CPU 卸载优化器 | True (显存不足时) |
| `enable_gradient_checkpointing` | 梯度检查点 | True |
| `use_liger` | 使用 Liger Kernel | True |
| `gpu_memory_utilization` | GPU 显存利用率 | 0.92 |

---

## 九、总结

verl 的 Rollout 和训练实现可以总结为：

1. **Rollout**: 
   - 异步模式下通过 `AgentLoopManager` 管理多个 `AgentLoopWorker`
   - 每个 Worker 运行一个状态机 (`ToolAgentLoop`) 处理多轮对话
   - 底层通过 HTTP 调用 SGLang Server 进行文本生成
   - 支持工具调用、多轮对话、提前终止等特性

2. **训练**:
   - 基于 Ray 实现分布式训练协调
   - 使用 FSDP 进行参数分片和梯度同步
   - 混合引擎模式下，同一组 GPU 交替进行训练和推理
   - 通过参数卸载、梯度检查点、混合精度等技术优化显存使用

3. **提速关键**:
   - FSDP 参数/优化器 CPU 卸载
   - 异步 Rollout 提高吞吐量
   - 训练-推理模式快速切换
   - 序列长度负载均衡
   - 高效的 Triton 内核 (Liger)

