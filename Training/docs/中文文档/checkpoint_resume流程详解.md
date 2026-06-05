# Checkpoint Resume 流程详解

本文档详细解析 VERL 框架中 checkpoint 恢复(resume)的完整代码逻辑。

## 目录
1. [整体流程概览](#整体流程概览)
2. [Trainer 层面的 Resume 逻辑](#trainer-层面的-resume-逻辑)
3. [Worker 层面的加载逻辑](#worker-层面的加载逻辑)
4. [FSDP CheckpointManager 的实现](#fsdp-checkpointmanager-的实现)
5. [完整调用链](#完整调用链)

---

## 整体流程概览

```
训练启动
    ↓
Trainer.fit()
    ↓
Trainer._load_checkpoint()  ← 确定 checkpoint 路径
    ↓
ActorRolloutRefWorker.load_checkpoint()  ← 加载 Actor
    ↓
CriticWorker.load_checkpoint() (如果使用 Critic)
    ↓
FSDPCheckpointManager.load_checkpoint()  ← 实际加载模型/优化器
    ↓
恢复训练 / 开始推理
```

---

## Trainer 层面的 Resume 逻辑

### 1. 入口: `RayPPOTrainer.fit()`

**文件位置**: `verl/trainer/ppo/ray_trainer.py:972`

```python
def fit(self):
    """训练主循环"""
    # 初始化 logger
    logger = Tracking(...)
    
    self.global_steps = 0
    
    # ⭐ 关键步骤：在训练前加载 checkpoint
    self._load_checkpoint()
    
    # 如果配置了 val_only=True，只做验证不训练
    if self.config.trainer.get("val_only", False):
        val_metrics = self._validate()
        return  # 直接返回，不进行训练
    
    # 继续训练循环
    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            # 训练逻辑...
```

### 2. 核心方法: `Trainer._load_checkpoint()`

**文件位置**: `verl/trainer/ppo/ray_trainer.py:833`

```python
def _load_checkpoint(self):
    """从 checkpoint 恢复训练状态"""
    
    # ========== Step 1: 检查 resume_mode ==========
    if self.config.trainer.resume_mode == "disable":
        return 0  # 不恢复，从头训练
    
    # ========== Step 2: 确定 checkpoint 路径 ==========
    checkpoint_folder = self.config.trainer.default_local_dir
    if not os.path.isabs(checkpoint_folder):
        working_dir = os.getcwd()
        checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
    
    # ========== Step 3: 根据 resume_mode 查找 checkpoint ==========
    if self.config.trainer.resume_mode == "auto":
        # 自动模式：查找最新的 checkpoint
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if global_step_folder is None:
            print("Training from scratch")
            return 0
            
    elif self.config.trainer.resume_mode == "resume_path":
        # 指定路径模式：使用用户指定的路径
        assert isinstance(self.config.trainer.resume_from_path, str), \
            "resume ckpt must be str type"
        assert "global_step_" in self.config.trainer.resume_from_path, \
            "resume ckpt must specify the global_steps"
        
        global_step_folder = self.config.trainer.resume_from_path
        if not os.path.isabs(global_step_folder):
            working_dir = os.getcwd()
            global_step_folder = os.path.join(working_dir, global_step_folder)
    
    print(f"Load from checkpoint folder: {global_step_folder}")
    
    # ========== Step 4: 解析并设置 global_steps ==========
    # 从路径中提取步数: "global_step_100" -> 100
    self.global_steps = int(global_step_folder.split("global_step_")[-1])
    print(f"Setting global step to {self.global_steps}")
    print(f"Resuming from {global_step_folder}")
    
    # ========== Step 5: 加载 Actor 和 Critic ==========
    actor_path = os.path.join(global_step_folder, "actor")
    critic_path = os.path.join(global_step_folder, "critic")
    
    # 调用 Worker 的 load_checkpoint 方法
    self.actor_rollout_wg.load_checkpoint(
        actor_path, 
        del_local_after_load=self.config.trainer.del_local_ckpt_after_load
    )
    
    if self.use_critic:
        self.critic_wg.load_checkpoint(
            critic_path, 
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
    
    # ========== Step 6: 恢复 DataLoader 状态 ==========
    dataloader_local_path = os.path.join(global_step_folder, "data.pt")
    if os.path.exists(dataloader_local_path):
        dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
        self.train_dataloader.load_state_dict(dataloader_state_dict)
    else:
        print(f"Warning: No dataloader state found at {dataloader_local_path}")
```

### 3. 辅助函数: `find_latest_ckpt_path()`

**文件位置**: `verl/utils/checkpoint/checkpoint_manager.py:167`

```python
def find_latest_ckpt_path(path, directory_format="global_step_{}"):
    """查找最新的 checkpoint 路径
    
    通过读取 tracker 文件（latest_checkpointed_iteration.txt）
    来找到最新保存的 checkpoint
    """
    if path is None:
        return None
    
    # 读取 tracker 文件
    tracker_file = get_checkpoint_tracker_filename(path)
    # tracker_file = "checkpoints/project/experiment/latest_checkpointed_iteration.txt"
    
    if not os.path.exists(tracker_file):
        print(f"Checkpoint tracker file does not exist: {tracker_file}")
        return None
    
    # 从 tracker 文件中读取 iteration 数字
    with open(tracker_file, "rb") as f:
        iteration = int(f.read().decode())  # 例如: 100
    
    # 构建 checkpoint 路径
    ckpt_path = os.path.join(path, directory_format.format(iteration))
    # 例如: "checkpoints/project/experiment/global_step_100"
    
    if not os.path.exists(ckpt_path):
        print("Checkpoint does not exist: %s", ckpt_path)
        return None
    
    print("Found checkpoint: %s", ckpt_path)
    return ckpt_path
```

---

## Worker 层面的加载逻辑

### 1. Worker 的 load_checkpoint 接口

**文件位置**: `verl/workers/fsdp_workers.py:1072`

```python
class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        """在所有 worker 上加载 checkpoint
        
        @register 装饰器确保这个方法会在所有分布式 worker 上执行
        dispatch_mode=ONE_TO_ALL 表示广播到所有节点
        """
        assert self._is_actor or (not self._is_actor and self._is_rollout), \
            "Checkpoint loading is only supported for Actor or standalone Rollout Workers"
        
        # ========== Step 1: 如果开启了参数 offload，先加载回 GPU ==========
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        
        # ========== Step 2: 调用 CheckpointManager 加载 ==========
        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, 
            hdfs_path=hdfs_path, 
            del_local_after_load=del_local_after_load
        )
        
        # ========== Step 3: 加载完成后重新 offload ==========
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)
```

---

## FSDP CheckpointManager 的实现

### 1. FSDPCheckpointManager.load_checkpoint()

**文件位置**: `verl/utils/checkpoint/fsdp_checkpoint_manager.py:98`

```python
class FSDPCheckpointManager(BaseCheckpointManager):
    """管理 FSDP 分片格式的 checkpoint"""
    
    def load_checkpoint(self, local_path: str, hdfs_path: str = None, 
                       del_local_after_load=False):
        """加载 FSDP checkpoint
        
        每个 rank 加载自己的分片数据:
        - model_world_size_8_rank_0.pt (rank 0)
        - model_world_size_8_rank_1.pt (rank 1)
        - ...
        """
        if local_path is None:
            return
        
        # ========== Step 1: 验证配置 ==========
        if self.should_load_model:
            assert self.model is not None, \
                "model must be provided when checkpoint_contents.load includes ['model']"
        if self.should_load_optimizer:
            assert self.optimizer is not None, \
                "optimizer must be provided when checkpoint_contents.load includes ['optimizer']"
        
        # ========== Step 2: 配置 FSDP state dict ==========
        state_dict_cfg = ShardedStateDictConfig(
            offload_to_cpu=True if is_cuda_available else False
        ) if self.should_load_model else None
        
        optim_cfg = ShardedOptimStateDictConfig(
            offload_to_cpu=True if is_cuda_available else False
        ) if self.should_load_optimizer else None
        
        # ========== Step 3: 加载模型和优化器 ==========
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, 
                               state_dict_cfg, optim_cfg):
            # 加载模型权重
            if self.should_load_model:
                remote_model_path = os.path.join(
                    local_path, 
                    f"model_world_size_{self.world_size}_rank_{self.rank}.pt"
                )
                local_model_path = copy_to_local(remote_model_path)
                model_state_dict = torch.load(local_model_path, weights_only=False)
                self.model.load_state_dict(model_state_dict)
                log_with_rank(f"Loaded model from {remote_model_path}", 
                             rank=self.rank, logger=logger)
            
            # 加载优化器状态
            if self.should_load_optimizer:
                remote_optim_path = os.path.join(
                    local_path, 
                    f"optim_world_size_{self.world_size}_rank_{self.rank}.pt"
                )
                local_optim_path = copy_to_local(remote_optim_path)
                optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
                self.optimizer.load_state_dict(optimizer_state_dict)
                log_with_rank(f"Loaded optimizer from {remote_optim_path}", 
                             rank=self.rank, logger=logger)
        
        # ========== Step 4: 加载额外状态（学习率调度器、RNG）==========
        if self.should_load_extra:
            remote_extra_state_path = os.path.join(
                local_path, 
                f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            local_extra_state_path = copy_to_local(remote_extra_state_path)
            extra_state_dict = torch.load(local_extra_state_path, weights_only=False)
            
            # 恢复随机数状态
            if "rng" in extra_state_dict:
                self.load_rng_state(extra_state_dict["rng"])
                log_with_rank(f"Loaded rng from {remote_extra_state_path}", 
                             rank=self.rank, logger=logger)
            
            # 恢复学习率调度器
            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
            if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                log_with_rank(f"Loaded lr_scheduler from {remote_extra_state_path}", 
                             rank=self.rank, logger=logger)
        
        # ========== Step 5: 清理和同步 ==========
        if self.rank == 0 and del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt file after loading failed, exception {e}",
                    rank=self.rank, logger=logger
                )
        
        # 等待所有 rank 完成加载
        torch.distributed.barrier()
```

---

## 完整调用链

### 训练启动时的 Resume 流程

```
1. bash agentloop_search_browse.sh
   ↓ 启动训练脚本
   
2. python3 -m verl.trainer.main_ppo
   ↓ 解析配置，初始化 Trainer
   
3. RayPPOTrainer.__init__()
   ↓ 初始化 Actor/Critic Workers
   
4. RayPPOTrainer.fit()
   ↓ 训练主循环
   
5. RayPPOTrainer._load_checkpoint()
   ├─ 检查 resume_mode (auto/disable/resume_path)
   ├─ 如果是 auto: find_latest_ckpt_path() 查找最新 checkpoint
   ├─ 如果是 resume_path: 使用 resume_from_path 指定的路径
   ├─ 解析 global_step 从路径名
   └─ 调用 Workers 加载
   
6. ActorRolloutRefWorker.load_checkpoint()  [在所有 ranks 上执行]
   ├─ 如果有 offload，先加载模型到 GPU
   ├─ 调用 CheckpointManager
   └─ 加载完成后重新 offload
   
7. FSDPCheckpointManager.load_checkpoint()  [每个 rank 加载自己的分片]
   ├─ 加载模型分片: model_world_size_8_rank_{rank}.pt
   ├─ 加载优化器分片: optim_world_size_8_rank_{rank}.pt
   ├─ 加载额外状态: extra_state_world_size_8_rank_{rank}.pt
   │   ├─ 学习率调度器状态
   │   └─ 随机数生成器状态
   └─ barrier 同步所有 ranks
   
8. 继续训练或验证
   ├─ 如果 val_only=True: 只做验证，然后退出
   └─ 否则: 从 global_steps+1 继续训练
```

### 推理时的 Resume 流程

```
1. bash inference_from_checkpoint.sh 100
   ↓ 指定 checkpoint step
   
2. python3 -m verl.trainer.main_ppo \
      trainer.resume_mode=resume_path \
      trainer.resume_from_path=/path/to/global_step_100 \
      trainer.val_only=True
   ↓
   
3. RayPPOTrainer.fit()
   ├─ _load_checkpoint() 
   │   └─ 加载 checkpoint (同上)
   ├─ _validate() 
   │   └─ 执行验证
   └─ return (因为 val_only=True)
```

---

## Checkpoint 目录结构

```
checkpoints/deepresearch/agentloop-dual-tools/
├── latest_checkpointed_iteration.txt  # 记录最新的 iteration: "100"
├── global_step_50/
│   ├── actor/
│   │   ├── model_world_size_8_rank_0.pt      # Rank 0 的模型分片
│   │   ├── model_world_size_8_rank_1.pt      # Rank 1 的模型分片
│   │   ├── ...
│   │   ├── optim_world_size_8_rank_0.pt      # Rank 0 的优化器状态
│   │   ├── optim_world_size_8_rank_1.pt
│   │   ├── ...
│   │   ├── extra_state_world_size_8_rank_0.pt  # Rank 0 的额外状态
│   │   ├── extra_state_world_size_8_rank_1.pt
│   │   ├── ...
│   │   ├── fsdp_config.json                  # FSDP 配置
│   │   └── huggingface/                      # HuggingFace 格式副本
│   │       ├── config.json
│   │       ├── tokenizer.json
│   │       ├── vocab.json
│   │       └── ...
│   ├── critic/ (如果使用)
│   └── data.pt  # DataLoader 状态
└── global_step_100/
    └── (同上结构)
```

---

## 关键配置参数

### 1. resume_mode

```yaml
trainer:
  resume_mode: auto  # 可选值: "auto" | "disable" | "resume_path"
```

- **auto**: 自动查找 `default_local_dir` 下最新的 checkpoint
- **disable**: 不恢复，从头训练
- **resume_path**: 使用 `resume_from_path` 指定的路径

### 2. resume_from_path

```yaml
trainer:
  resume_mode: resume_path
  resume_from_path: /path/to/global_step_100
```

仅在 `resume_mode=resume_path` 时使用。

### 3. val_only

```yaml
trainer:
  val_only: True  # 只做推理，不训练
```

### 4. default_local_dir

```yaml
trainer:
  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
```

自动模式下查找 checkpoint 的目录。

---

## 使用示例

### 示例 1: 从最新 checkpoint 继续训练

```bash
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_config \
    trainer.resume_mode=auto \
    trainer.default_local_dir=checkpoints/my_project/exp1
```

### 示例 2: 从指定 checkpoint 继续训练

```bash
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_config \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=checkpoints/my_project/exp1/global_step_100
```

### 示例 3: 从 checkpoint 进行推理（只验证）

```bash
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_config \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=checkpoints/my_project/exp1/global_step_100 \
    trainer.val_only=True \
    trainer.val_before_train=True
```

---

## 常见问题

### Q1: 为什么 actor_rollout_ref.model.path 不能直接指向 checkpoint？

**A**: `actor_rollout_ref.model.path` 用于初始化模型，需要标准的 HuggingFace 格式。Checkpoint 是 FSDP 分片格式，需要通过 `resume_from_path` 加载。

### Q2: 可以只加载模型权重，不加载优化器吗？

**A**: 可以，通过配置 `checkpoint_config`:

```yaml
actor_rollout_ref:
  actor:
    checkpoint:
      load: ['model']  # 只加载模型
      save: ['model', 'optimizer', 'extra']  # 保存时还是保存全部
```

### Q3: huggingface 子目录中的模型可以直接用吗？

**A**: 可以用于推理，但会丢失优化器状态和训练元数据：

```bash
actor_rollout_ref.model.path=/path/to/checkpoint/actor/huggingface
```

不推荐用于继续训练。

---

## 总结

Resume 流程的关键点：

1. **Trainer 层**: 负责确定 checkpoint 路径，协调加载
2. **Worker 层**: 处理分布式加载，管理 offload
3. **CheckpointManager 层**: 实际加载 FSDP 分片数据
4. **每个 rank 独立加载**: 自己的模型/优化器分片
5. **Barrier 同步**: 确保所有 ranks 完成后再继续

推理时只需设置 `val_only=True`，系统会：
- 加载 checkpoint
- 执行验证
- 直接退出，不进行训练

