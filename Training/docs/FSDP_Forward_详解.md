# FSDP Forward 计算详解：从 input_ids 到 logits

本文档详细解释在 GRPO 训练中，FSDP 模型 forward 的完整计算过程，包括每一步的 tensor 维度、计算原理和显存占用。

---

## 具体示例参数

以 **Qwen2.5-7B** 模型为例：

```python
# ========== 模型配置 ==========
hidden_size = 3584          # 隐藏层维度
num_hidden_layers = 28      # Transformer 层数
num_attention_heads = 28    # 注意力头数
head_dim = 128             # 每个头的维度 (hidden_size / num_attention_heads)
intermediate_size = 18944   # FFN 中间层维度
vocab_size = 152064         # 词表大小

# ========== 输入数据 ==========
batch_size = 16             # 批大小
prompt_length = 512         # Prompt 长度
response_length = 128       # 响应长度
seq_length = 640           # 总序列长度 = 512 + 128

# ========== 输入 Tensor ==========
input_ids: torch.LongTensor = (16, 640)      # Token IDs
attention_mask: torch.Tensor = (16, 640)     # 注意力 mask（1=有效, 0=padding）
position_ids: torch.LongTensor = (16, 640)   # 位置编码 [0, 1, 2, ..., 639]
```

---

## Step 1: Embedding Layer（词嵌入层）

### 代码位置
`transformers/models/qwen2/modeling_qwen2.py: Qwen2Model.forward()`

### 计算过程

```python
class Qwen2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 词嵌入矩阵
        self.embed_tokens = nn.Embedding(
            num_embeddings=152064,  # vocab_size
            embedding_dim=3584       # hidden_size
        )
        # → 参数: (152064, 3584)
        # → 大小: 152064 * 3584 * 2 bytes (bf16) ≈ 1.09 GB

    def forward(self, input_ids, attention_mask, position_ids, ...):
        # ========== FSDP All-Gather ==========
        # 在 FSDP 模式下，embed_tokens.weight 被分片存储
        # Forward 时自动 All-Gather 聚合完整参数（临时）
        # GPU 0: 1.09 GB / 8 ≈ 136 MB (分片)
        # All-Gather 后: 1.09 GB (完整)

        # ========== Token Embedding ==========
        inputs_embeds = self.embed_tokens(input_ids)
        # input_ids: (16, 640)
        # → inputs_embeds: (16, 640, 3584)
        #
        # 计算原理：
        # 对于每个 token ID，从 embed_tokens.weight 中查表
        # inputs_embeds[i, j, :] = embed_tokens.weight[input_ids[i, j], :]
        #
        # 具体示例：
        # input_ids[0, 0] = 12345 (某个 token ID)
        # → inputs_embeds[0, 0, :] = embed_tokens.weight[12345, :]
        #                           = [0.12, -0.34, 0.56, ..., 0.78] (3584 维向量)

        # ========== FSDP 释放参数 ==========
        # Forward 完成后，FSDP 自动释放聚合的完整参数
        # 只保留分片（136 MB）

        hidden_states = inputs_embeds  # (16, 640, 3584)
        return hidden_states
```

### 显存占用

```
┌─────────────────────────────────────────────┐
│         Embedding Layer 显存占用            │
└─────────────────────────────────────────────┘

输入:
├── input_ids: 16 * 640 * 4 bytes (int32) = 40 KB

参数（FSDP All-Gather，临时）:
├── embed_tokens.weight (完整): 1.09 GB
├── 分片后 (每个 GPU): 136 MB
└── Forward 后释放，只保留分片: 136 MB

输出（激活值，保留用于 backward）:
└── hidden_states: 16 * 640 * 3584 * 2 bytes (bf16) = 73.7 MB
```

### 具体数值示例

```python
# 假设输入
input_ids[0, 0:5] = [198, 12345, 6789, 54321, 999]

# Embedding 后
inputs_embeds[0, 0, :] = embed_tokens.weight[198, :]
# = [0.0234, -0.1234, 0.5678, ..., -0.0987]  (3584 维)

inputs_embeds[0, 1, :] = embed_tokens.weight[12345, :]
# = [-0.3456, 0.7890, -0.2345, ..., 0.4567]  (3584 维)

# ... 依此类推

# 最终
inputs_embeds[0, :, :].shape = (640, 3584)  # 640 个 token，每个 3584 维
```

---

## Step 2: Transformer Layers（28 层）

每一层的计算流程相同，这里以 **第 0 层** 为例。

### 代码位置
`transformers/models/qwen2/modeling_qwen2.py: Qwen2DecoderLayer.forward()`

---

### 2.1 Pre-Norm (RMSNorm)

```python
class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # RMSNorm 只有一个可学习的 scale 参数
        self.input_layernorm = Qwen2RMSNorm(
            hidden_size=3584,
            eps=1e-6
        )
        # → 参数: (3584,)
        # → 大小: 3584 * 2 bytes (bf16) = 7 KB

    def forward(self, hidden_states, attention_mask, position_ids, ...):
        # 输入: hidden_states = (16, 640, 3584)

        residual = hidden_states  # 保存用于残差连接

        # ========== RMSNorm ==========
        hidden_states = self.input_layernorm(hidden_states)
        # → hidden_states: (16, 640, 3584)
        #
        # RMSNorm 计算公式：
        # 1. 计算 RMS（Root Mean Square）
        #    variance = mean(hidden_states^2) + eps
        #    RMS = sqrt(variance)
        # 2. 归一化并应用可学习的 scale
        #    output = hidden_states / RMS * self.weight
        #
        # 具体示例（某个 token 的归一化）：
        # hidden_states[0, 5, :] = [0.5, -0.3, 0.8, ..., 0.2]  (3584 维)
        # variance = mean([0.5^2, (-0.3)^2, 0.8^2, ..., 0.2^2]) + 1e-6
        #          = 0.25 + 0.09 + 0.64 + ... + 0.04) / 3584 + 1e-6
        #          ≈ 0.15
        # RMS = sqrt(0.15) ≈ 0.387
        # output = [0.5, -0.3, 0.8, ..., 0.2] / 0.387 * self.weight
        #        = [1.29 * w0, -0.77 * w1, 2.07 * w2, ..., 0.52 * w3583]
```

---

### 2.2 Self-Attention

```python
class Qwen2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = 3584
        self.num_heads = 28
        self.head_dim = 128  # 3584 / 28

        # Q, K, V 投影矩阵
        self.q_proj = nn.Linear(3584, 3584, bias=True)
        self.k_proj = nn.Linear(3584, 3584, bias=True)
        self.v_proj = nn.Linear(3584, 3584, bias=True)
        # 每个参数: (3584, 3584) = 25.7 MB (bf16)

        # 输出投影矩阵
        self.o_proj = nn.Linear(3584, 3584, bias=False)
        # 参数: (3584, 3584) = 25.7 MB

        # 总计: 4 * 25.7 MB ≈ 102.8 MB (完整)
        #      102.8 MB / 8 GPUs ≈ 12.9 MB (分片)
```

#### 2.2.1 计算 Q, K, V

```python
def forward(self, hidden_states, attention_mask, position_ids, ...):
    # 输入: hidden_states = (16, 640, 3584)

    # ========== FSDP All-Gather ==========
    # 聚合 q_proj, k_proj, v_proj, o_proj 的完整参数
    # 临时增加: 102.8 MB

    # ========== 计算 Q, K, V ==========
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    # hidden_states: (16, 640, 3584)
    # → query_states: (16, 640, 3584)
    # → key_states: (16, 640, 3584)
    # → value_states: (16, 640, 3584)
    #
    # 矩阵乘法原理（以 q_proj 为例）：
    # query_states = hidden_states @ q_proj.weight.T + q_proj.bias
    # (16, 640, 3584) @ (3584, 3584) → (16, 640, 3584)
    #
    # 具体计算（某个位置）：
    # query_states[0, 5, k] = sum(hidden_states[0, 5, i] * q_proj.weight[k, i]
    #                              for i in range(3584)) + q_proj.bias[k]
```

#### 2.2.2 Reshape 为多头

```python
    # ========== Reshape 为多头 ==========
    bsz, q_len, _ = hidden_states.size()  # 16, 640, 3584

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    # (16, 640, 3584) → (16, 640, 28, 128)

    query_states = query_states.transpose(1, 2)
    # (16, 640, 28, 128) → (16, 28, 640, 128)
    #
    # 原理：将 hidden_size 维度拆分为 num_heads 个 head_dim
    # 每个注意力头独立计算，最后合并

    key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # → query_states: (16, 28, 640, 128)
    # → key_states: (16, 28, 640, 128)
    # → value_states: (16, 28, 640, 128)
```

#### 2.2.3 应用 RoPE（Rotary Position Embedding）

```python
    # ========== 计算 RoPE ==========
    cos, sin = self.rotary_emb(value_states, position_ids)
    # position_ids: (16, 640) = [[0, 1, 2, ..., 639], ...]
    # → cos, sin: (16, 640, 128)  # 每个位置的旋转矩阵

    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    # → query_states: (16, 28, 640, 128) [已编码位置信息]
    # → key_states: (16, 28, 640, 128) [已编码位置信息]
    #
    # RoPE 原理：
    # 通过旋转变换将位置信息编码到 Q 和 K 中
    # 对于位置 m 和 n，内积满足：
    # <RoPE(q, m), RoPE(k, n)> = f(q, k, m-n)
    # 即内积只依赖于相对位置 m-n
    #
    # 具体计算（简化版）：
    # θ_i = 10000^(-2i/d)  # d = head_dim = 128
    # RoPE(x, m) = [x[0] * cos(m*θ0) - x[1] * sin(m*θ0),
    #               x[0] * sin(m*θ0) + x[1] * cos(m*θ0),
    #               x[2] * cos(m*θ1) - x[3] * sin(m*θ1),
    #               x[2] * sin(m*θ1) + x[3] * cos(m*θ1),
    #               ...]
```

#### 2.2.4 计算 Attention Scores

```python
    # ========== 计算 Attention Scores ==========
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    # query_states: (16, 28, 640, 128)
    # key_states.transpose(2, 3): (16, 28, 128, 640)
    # → attn_weights: (16, 28, 640, 640)
    #
    # 原理：计算 Q 和 K 的点积，衡量每个 token 之间的相关性
    # attn_weights[i, h, j, k] = sum(query_states[i, h, j, d] * key_states[i, h, k, d]
    #                                 for d in range(128))
    #
    # 具体示例（某个头，某两个位置）：
    # query_states[0, 0, 5, :] = [0.1, -0.2, 0.3, ..., 0.4]  (128 维)
    # key_states[0, 0, 10, :] = [-0.1, 0.3, 0.2, ..., -0.2]  (128 维)
    # attn_weights[0, 0, 5, 10] = 0.1*(-0.1) + (-0.2)*0.3 + 0.3*0.2 + ... + 0.4*(-0.2)
    #                            = -0.01 - 0.06 + 0.06 + ... - 0.08
    #                            ≈ 2.34

    # ========== Scaling ==========
    attn_weights = attn_weights / math.sqrt(self.head_dim)
    # → attn_weights: (16, 28, 640, 640)
    #
    # 原理：缩放防止点积过大导致 softmax 梯度消失
    # attn_weights = attn_weights / sqrt(128) ≈ attn_weights / 11.31
    # 例如：2.34 / 11.31 ≈ 0.207
```

#### 2.2.5 应用 Attention Mask

```python
    # ========== Causal Mask ==========
    # attention_mask: (16, 1, 640, 640)
    # 对于 Causal LM，mask 是下三角矩阵：
    # [[0, -inf, -inf, -inf, ...],
    #  [0,    0, -inf, -inf, ...],
    #  [0,    0,    0, -inf, ...],
    #  [0,    0,    0,    0, ...],
    #  ...]
    #
    # 原理：
    # 1. 当前 token 只能看到之前的 token（Causal Attention）
    # 2. Padding token 不参与计算（mask = -inf）

    attn_weights = attn_weights + attention_mask
    # → attn_weights: (16, 28, 640, 640)
    #
    # 具体示例：
    # 原始 attn_weights[0, 0, 5, 10] = 0.207
    # mask[0, 0, 5, 10] = -inf (因为 10 > 5)
    # → attn_weights[0, 0, 5, 10] = 0.207 + (-inf) = -inf
    #
    # attn_weights[0, 0, 5, 3] = 0.156
    # mask[0, 0, 5, 3] = 0 (因为 3 < 5)
    # → attn_weights[0, 0, 5, 3] = 0.156 + 0 = 0.156
```

#### 2.2.6 Softmax

```python
    # ========== Softmax ==========
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
    # → attn_weights: (16, 28, 640, 640)
    #
    # 原理：将 attention scores 转换为概率分布（每行和为 1）
    # softmax(x_i) = exp(x_i) / sum(exp(x_j) for j)
    #
    # 具体示例（第 0 个样本，第 0 个头，第 5 个位置）：
    # 原始 scores[0, 0, 5, :] = [0.12, 0.08, -0.05, 0.15, 0.23, 0.18, -inf, -inf, ...]
    #                             ↑     ↑      ↑     ↑     ↑     ↑      ↑      ↑
    #                          pos 0   1      2     3     4     5      6      7  ...
    #
    # Softmax 后:
    # attn_weights[0, 0, 5, :] = [0.18, 0.17, 0.15, 0.19, 0.21, 0.19, 0.0, 0.0, ...]
    #                             └─────────────── 和为 1.0 ──────────────┘
    #
    # 注意：-inf 位置 softmax 后变为 0（不可见的 token）

    attn_weights = attn_weights.to(query_states.dtype)  # 转回 bf16
```

#### 2.2.7 Attention Output

```python
    # ========== Attention Output ==========
    attn_output = torch.matmul(attn_weights, value_states)
    # attn_weights: (16, 28, 640, 640)
    # value_states: (16, 28, 640, 128)
    # → attn_output: (16, 28, 640, 128)
    #
    # 原理：对 Value 进行加权求和，权重为 attention scores
    # attn_output[i, h, j, d] = sum(attn_weights[i, h, j, k] * value_states[i, h, k, d]
    #                                for k in range(640))
    #
    # 具体示例（某个头，某个位置）：
    # attn_weights[0, 0, 5, :] = [0.18, 0.17, 0.15, 0.19, 0.21, 0.19, 0, 0, ...]
    # value_states[0, 0, 0:6, 0] = [0.5, -0.3, 0.2, 0.8, -0.1, 0.6, ...]
    # attn_output[0, 0, 5, 0] = 0.18*0.5 + 0.17*(-0.3) + 0.15*0.2 + 0.19*0.8 + 0.21*(-0.1) + 0.19*0.6
    #                         = 0.09 - 0.051 + 0.03 + 0.152 - 0.021 + 0.114
    #                         ≈ 0.314
```

#### 2.2.8 Reshape 回单头

```python
    # ========== Reshape ==========
    attn_output = attn_output.transpose(1, 2).contiguous()
    # (16, 28, 640, 128) → (16, 640, 28, 128)

    attn_output = attn_output.view(bsz, q_len, self.hidden_size)
    # (16, 640, 28, 128) → (16, 640, 3584)
    #
    # 原理：将多头合并回 hidden_size 维度
```

#### 2.2.9 Output Projection

```python
    # ========== Output Projection ==========
    attn_output = self.o_proj(attn_output)
    # attn_output @ o_proj.weight.T
    # (16, 640, 3584) @ (3584, 3584) → (16, 640, 3584)

    # ========== Residual Connection ==========
    hidden_states = residual + attn_output
    # → hidden_states: (16, 640, 3584)

    # ========== FSDP 释放参数 ==========
    # 释放 Self-Attention 的完整参数（102.8 MB）
    # 只保留分片（12.9 MB）
```

### Self-Attention 显存占用

```
┌─────────────────────────────────────────────┐
│        Self-Attention 显存占用              │
└─────────────────────────────────────────────┘

参数（FSDP All-Gather，临时）:
├── q_proj, k_proj, v_proj, o_proj: 102.8 MB (完整)
├── Forward 后释放，只保留分片: 12.9 MB

激活值（保留用于 backward）:
├── query/key/value_states: 3 * (16*640*3584*2) = 221.1 MB
├── attn_weights: 16*28*640*640*2 = 366 MB
└── attn_output: 16*640*3584*2 = 73.7 MB

峰值: 102.8 MB (参数) + 660.8 MB (激活值) ≈ 763.6 MB
```

---

### 2.3 Post-Attention Norm

```python
    # ========== Post-Attention Norm ==========
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    # → hidden_states: (16, 640, 3584)
    # 计算原理同 Pre-Norm
```

---

### 2.4 Feed-Forward Network (FFN)

```python
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = 3584
        self.intermediate_size = 18944

        # Up projection 和 Gate projection
        self.gate_proj = nn.Linear(3584, 18944, bias=False)
        self.up_proj = nn.Linear(3584, 18944, bias=False)
        # 每个参数: (3584, 18944) = 136 MB (bf16)

        # Down projection
        self.down_proj = nn.Linear(18944, 3584, bias=False)
        # 参数: (18944, 3584) = 136 MB

        # 总计: 408 MB (完整), 51 MB (分片)

    def forward(self, hidden_states):
        # 输入: hidden_states = (16, 640, 3584)

        # ========== FSDP All-Gather ==========
        # 聚合 gate_proj, up_proj, down_proj 的完整参数
        # 临时增加: 408 MB

        # ========== Up Projection + Gate ==========
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        # hidden_states: (16, 640, 3584)
        # → gate: (16, 640, 18944)
        # → up: (16, 640, 18944)
        #
        # 矩阵乘法:
        # gate = hidden_states @ gate_proj.weight.T
        # (16, 640, 3584) @ (18944, 3584).T → (16, 640, 18944)

        # ========== Activation (SwiGLU) ==========
        hidden_states = self.act_fn(gate) * up
        # → hidden_states: (16, 640, 18944)
        #
        # SwiGLU 原理：
        # SwiGLU(x) = Swish(gate) * up
        # Swish(x) = x * sigmoid(x)
        #
        # 具体计算（某个位置，某个维度）：
        # gate[0, 5, 100] = 2.3
        # up[0, 5, 100] = -0.8
        # Swish(2.3) = 2.3 * sigmoid(2.3) = 2.3 * 0.909 ≈ 2.09
        # hidden_states[0, 5, 100] = 2.09 * (-0.8) ≈ -1.67

        # ========== Down Projection ==========
        hidden_states = self.down_proj(hidden_states)
        # (16, 640, 18944) @ (3584, 18944).T → (16, 640, 3584)

        # ========== FSDP 释放参数 ==========
        # 释放 FFN 的完整参数（408 MB）
        # 只保留分片（51 MB）

        return hidden_states  # (16, 640, 3584)
```

### FFN 显存占用

```
┌─────────────────────────────────────────────┐
│         FFN 显存占用                        │
└─────────────────────────────────────────────┘

参数（FSDP All-Gather，临时）:
├── gate_proj, up_proj, down_proj: 408 MB (完整)
├── Forward 后释放，只保留分片: 51 MB

激活值（保留用于 backward）:
├── gate, up: 2 * (16*640*18944*2) = 776 MB
├── 中间激活值: 16*640*18944*2 = 388 MB
└── 输出: 16*640*3584*2 = 73.7 MB

峰值: 408 MB (参数) + 1237.7 MB (激活值) ≈ 1.6 GB
```

---

### 单层 Transformer 总结

```python
    # ========== Residual Connection ==========
    hidden_states = residual + hidden_states
    # → hidden_states: (16, 640, 3584)

    # 传递给下一层
    return hidden_states
```

### 单层显存时间线

```
┌─────────────────────────────────────────────┐
│    单层 Transformer 显存变化（GPU 0）       │
└─────────────────────────────────────────────┘

时刻 0ms: 输入
├── hidden_states: 73.7 MB
└── 总计: 73.7 MB

时刻 0-2ms: Self-Attention
├── FSDP All-Gather: +102.8 MB (临时)
├── 激活值: +660.8 MB
├── 峰值: 73.7 + 102.8 + 660.8 = 837.3 MB
├── 释放参数: -102.8 MB
└── 当前: 73.7 + 660.8 = 734.5 MB

时刻 2-5ms: FFN
├── FSDP All-Gather: +408 MB (临时)
├── 激活值: +1237.7 MB
├── 峰值: 734.5 + 408 + 1237.7 = 2.38 GB ← **单层峰值**
├── 释放参数: -408 MB
├── 释放部分激活值: -1164 MB
└── 当前: 73.7 MB (输出)

时刻 5ms: 输出
└── hidden_states: 73.7 MB (传递给下一层)
```

**注意**：实际训练中，为了 backward，会保留部分激活值，总激活值约 **~500 MB per layer**。

---

## Step 3: Output Layer (LM Head)

### 代码位置
`transformers/models/qwen2/modeling_qwen2.py: Qwen2ForCausalLM.forward()`

```python
class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen2Model(config)  # 包含 Embedding + 28 层 Transformer

        # LM Head (输出层)
        self.lm_head = nn.Linear(
            in_features=3584,      # hidden_size
            out_features=152064,   # vocab_size
            bias=False
        )
        # → 参数: (152064, 3584)
        # → 大小: 152064 * 3584 * 2 bytes (bf16) ≈ 1.09 GB

    def forward(self, input_ids, attention_mask, position_ids, ...):
        # 经过 28 层 Transformer 后
        # hidden_states: (16, 640, 3584)

        # ========== Final Norm ==========
        hidden_states = self.model.norm(hidden_states)
        # → hidden_states: (16, 640, 3584)

        # ========== FSDP All-Gather ==========
        # 聚合 lm_head.weight 的完整参数
        # 临时增加: 1.09 GB

        # ========== LM Head ==========
        logits = self.lm_head(hidden_states)
        # hidden_states @ lm_head.weight.T
        # (16, 640, 3584) @ (152064, 3584).T → (16, 640, 152064)
        #
        # 原理：将隐藏状态映射到词表空间
        # logits[i, j, k] 表示：
        #   - 第 i 个样本
        #   - 第 j 个位置
        #   - 预测为第 k 个 token 的分数（未归一化）
        #
        # 具体计算（某个位置）：
        # logits[0, 5, k] = sum(hidden_states[0, 5, d] * lm_head.weight[k, d]
        #                       for d in range(3584))
        #
        # 例如：
        # logits[0, 5, 198] = 2.34 (token 198 的分数)
        # logits[0, 5, 12345] = -0.56 (token 12345 的分数)
        # logits[0, 5, 6789] = 3.78 (token 6789 的分数)
        # ...

        # ========== FSDP 释放参数 ==========
        # 释放 lm_head.weight 的完整参数（1.09 GB）
        # 只保留分片（136 MB）

        return logits  # (16, 640, 152064)
```

### LM Head 显存占用

```
┌─────────────────────────────────────────────┐
│         LM Head 显存占用                    │
└─────────────────────────────────────────────┘

参数（FSDP All-Gather，临时）:
├── lm_head.weight: 1.09 GB (完整)
├── Forward 后释放，只保留分片: 136 MB

输出（激活值）:
└── logits: 16*640*152064*2 = 3.12 GB ← **最大激活值！**
```

---

## Step 4: 提取 Response 部分的 Logits

```python
# 回到 verl/workers/actor/dp_actor.py:_forward_micro_batch

# logits: (16, 640, 152064)
prompt_length = 512
response_length = 128

# 提取响应部分（去掉最后一个 token）
logits = logits[:, -response_length-1:-1, :]
# logits[:, 512:640-1, :] → (16, 127, 152064)
#
# 实际代码中，slice 会包含第一个 response token
# 所以是 (16, 128, 152064)

# 原理：
# - 我们只需要响应部分的 logits
# - 去掉最后一个 token 是因为要用 logits[t] 预测 response[t+1]
# - Prompt 部分的 logits 不需要（不参与训练）
```

---

## Step 5: 计算 Log Probabilities

### 代码位置
`verl/utils/torch_functional.py: logprobs_from_logits()`

```python
def logprobs_from_logits(logits, labels, inplace_backward=True):
    """
    Args:
        logits: (batch_size, seq_length, vocab_size) = (16, 128, 152064)
        labels: (batch_size, seq_length) = (16, 128)  # 实际的 response tokens

    Returns:
        log_probs: (batch_size, seq_length) = (16, 128)
    """
    # ========== 步骤 5.1: Log Softmax ==========
    log_probs_all = F.log_softmax(logits, dim=-1)
    # logits: (16, 128, 152064)
    # → log_probs_all: (16, 128, 152064)
    #
    # 原理：将 logits 转换为 log probabilities
    # log_softmax(x_i) = log(exp(x_i) / sum(exp(x_j)))
    #                  = x_i - log(sum(exp(x_j)))
    #
    # 数值稳定版本（避免 overflow）：
    # max_val = max(logits)
    # log_softmax(x_i) = x_i - max_val - log(sum(exp(x_j - max_val)))
    #
    # 具体示例（某个位置）：
    # logits[0, 5, :] = [2.34, -0.56, 3.78, 1.23, ..., -2.45]  (152064 个值)
    # max_val = 3.78
    # log_sum_exp = log(exp(2.34-3.78) + exp(-0.56-3.78) + exp(3.78-3.78) + ...)
    #             = log(exp(-1.44) + exp(-4.34) + exp(0) + ...)
    #             ≈ log(0.237 + 0.013 + 1.0 + ...)
    #             ≈ log(1.5)
    #             ≈ 0.405
    # log_probs_all[0, 5, 0] = 2.34 - 3.78 - 0.405 = -1.845
    # log_probs_all[0, 5, 2] = 3.78 - 3.78 - 0.405 = -0.405 (最高概率)

    # ========== 步骤 5.2: 选择对应 label 的 log_prob ==========
    # 我们只需要真实 token 对应的 log probability
    log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1))
    # log_probs_all: (16, 128, 152064)
    # labels.unsqueeze(-1): (16, 128, 1)
    # → log_probs: (16, 128, 1)

    log_probs = log_probs.squeeze(-1)
    # → log_probs: (16, 128)
    #
    # 原理：
    # 对于每个位置，选择真实 token 的 log probability
    # 例如：如果 labels[0, 5] = 6789（token ID）
    #      则 log_probs[0, 5] = log_probs_all[0, 5, 6789]
    #                          = -2.13 (假设)

    return log_probs  # (16, 128) ← **需要梯度！**
```

### 具体数值示例

```python
# 假设某个位置的计算
i, j = 0, 5  # 第 0 个样本，第 5 个位置

# ========== Logits ==========
logits[i, j, :]: (152064,)
# [2.34, -0.56, 3.78, 1.23, ..., -2.45]
#   ↑      ↑      ↑      ↑          ↑
# ID=0   ID=1   ID=2   ID=3   ... ID=152063

# ========== Log Softmax ==========
log_probs_all[i, j, :]: (152064,)
# [-1.845, -4.745, -0.405, -2.955, ..., -6.635]
#
# 注意：exp(log_probs_all).sum() = 1.0
# 即：exp(-1.845) + exp(-4.745) + exp(-0.405) + ... = 1.0

# ========== 选择真实 Token ==========
# 假设真实 token 是 labels[i, j] = 6789
log_probs[i, j] = log_probs_all[i, j, 6789]
# = -2.13 (假设)
#
# 含义：
# - 模型预测 token 6789 的 log probability 是 -2.13
# - 对应的概率是 exp(-2.13) ≈ 0.12 (12%)
# - 越接近 0 越好（概率越高）
# - 越接近 -inf 越差（概率越低）
```

---

## 完整 Forward 流程的显存时间线

```
┌────────────────────────────────────────────────────────────┐
│           FSDP Forward 显存变化（单个 GPU）                │
└────────────────────────────────────────────────────────────┘

输入数据:
├── input_ids: (16, 640) = 40 KB
├── attention_mask: (16, 640) = 80 KB
├── position_ids: (16, 640) = 40 KB
└── 总计: 160 KB

─────────────────────────────────────────────────────────────

时刻 0-5ms: Embedding Layer
├── FSDP All-Gather: +1.09 GB (临时)
├── 计算 embedding: +73.7 MB (激活值)
├── 峰值: 1.09 GB + 73.7 MB ≈ 1.16 GB
├── 释放参数: -1.09 GB
└── 当前: 73.7 MB

─────────────────────────────────────────────────────────────

时刻 5-200ms: Transformer Layers (28 层)

对于每一层（约 7ms per layer）:
├── FSDP All-Gather 当前层参数: +510.8 MB (临时)
├── Self-Attention: 激活值 +660.8 MB
├── FFN: 激活值 +1237.7 MB
├── 单层峰值: 510.8 + 1237.7 + 73.7 ≈ 1.82 GB
├── 释放当前层参数: -510.8 MB
├── 释放部分激活值: -约 1700 MB
└── 保留输出: 73.7 MB (传递给下一层)

累积激活值（所有层，用于 backward）:
└── 约 28 * 50 MB ≈ 1.4 GB

─────────────────────────────────────────────────────────────

时刻 200-210ms: LM Head
├── FSDP All-Gather: +1.09 GB (临时)
├── 计算 logits: +3.12 GB (激活值) ← **最大激活值！**
├── 峰值: 1.4 + 1.09 + 3.12 ≈ 5.61 GB ← **全局峰值**
├── 释放参数: -1.09 GB
└── 当前: 1.4 + 3.12 = 4.52 GB

─────────────────────────────────────────────────────────────

时刻 210-215ms: 计算 log_probs
├── log_softmax: 计算在 logits 上
├── gather: 提取对应 token 的 log_prob
├── 可能释放完整 logits: -3.12 GB
├── log_probs: (16, 128) = 4 KB ← **需要梯度！**
└── 当前: 1.4 GB (激活值) + 4 KB (log_probs) ≈ 1.4 GB

─────────────────────────────────────────────────────────────

最终状态:
├── 激活值（用于 backward）: 约 1.4-2 GB
├── log_probs（需要梯度）: 4 KB
└── 总计: 约 1.4-2 GB

注意：
1. 峰值出现在 LM Head 计算 logits 时（约 5.6 GB）
2. 实际显存占用取决于 activation checkpointing 的使用
3. 如果启用 activation checkpointing，可减少约 50% 激活值显存
