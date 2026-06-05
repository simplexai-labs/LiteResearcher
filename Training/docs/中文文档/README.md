# VERL 中文文档索引

本目录包含所有verl相关的中文使用文档和说明。

## 📚 文档列表

### 1. [benchmark推理指南.md](./benchmark推理指南.md)
**用途**: 如何对多个benchmark进行推理测试（不训练）

**内容**:
- 6个benchmark数据的处理和结构说明
- 推理脚本使用方法
- 合并数据集和单独推理的区别
- 可视化轨迹的方法
- 性能估算和故障排查

**适用场景**:
- 需要在GAIA、GPQA、HLE、WebWalkerQA、Browsecomp、Xbench等benchmark上测试模型
- 只做推理不训练

---

### 2. [reward奖励计算说明.md](./reward奖励计算说明.md)
**用途**: 理解verl如何根据data_source选择和计算reward

**内容**:
- Reward函数选择机制详解
- `default_compute_score()` 工作原理
- 如何添加新的data_source
- Exact Match (EM) 评分逻辑
- 答案归一化处理
- 自定义reward函数的方法

**适用场景**:
- 需要理解reward是如何计算的
- 添加新的benchmark或数据集
- 调试reward相关问题

---

### 3. [轨迹元数据实现.md](./轨迹元数据实现.md)
**用途**: 如何在轨迹中添加assistant_turns、user_turns、pred_ans等元数据

**内容**:
- 元数据字段的添加位置
- 数据流向追踪（agent_loop → reward → trainer → dump）
- 修改的代码文件说明
- 验证方法

**适用场景**:
- 需要在轨迹文件中记录更多信息
- 分析对话轮数和预测答案
- 调试轨迹数据

---

### 4. [验证流程说明.md](./验证流程说明.md)
**用途**: 详细解释verl的validation验证流程

**内容**:
- Validation vs Training的区别
- val_before_train工作原理
- 验证数据的加载和处理
- 多轮对话验证的特殊处理
- 验证指标的计算

**适用场景**:
- 理解validation是如何工作的
- 调试验证相关问题
- 需要自定义验证流程

---

### 5. [verl验证指南.md](./verl验证指南.md)
**用途**: Verl验证功能的完整使用指南

**内容**:
- 验证配置详解
- 验证触发时机
- 验证指标解读
- 常见问题和解决方案
- 最佳实践

**适用场景**:
- 需要配置和使用validation功能
- 理解验证输出的指标

---

### 6. [多数投票说明.md](./多数投票说明.md)
**用途**: 如何使用majority voting提高准确率

**内容**:
- Majority voting工作原理
- 如何配置和使用
- 投票策略说明
- 实现细节

**适用场景**:
- 需要使用多次采样提高准确率
- 理解majority voting机制

---

### 7. [配置文件指南.md](./配置文件指南.md)
**用途**: Verl配置系统详解

**内容**:
- Hydra配置系统介绍
- 配置文件结构和层级
- 常用配置项说明
- 配置覆盖方法
- 配置文件生成

**适用场景**:
- 需要修改训练/推理配置
- 理解配置系统工作原理
- 创建新的实验配置

---

## 📖 快速导航

### 按使用场景分类

#### 🚀 快速开始
1. [benchmark推理指南.md](./benchmark推理指南.md) - 开始推理测试
2. [配置文件指南.md](./配置文件指南.md) - 理解和修改配置

#### 🔧 深入理解
1. [reward奖励计算说明.md](./reward奖励计算说明.md) - Reward机制
2. [验证流程说明.md](./验证流程说明.md) - Validation机制
3. [轨迹元数据实现.md](./轨迹元数据实现.md) - 轨迹数据结构

#### 📊 高级功能
1. [多数投票说明.md](./多数投票说明.md) - Majority voting
2. [verl验证指南.md](./verl验证指南.md) - 验证最佳实践

### 按文档类型分类

#### 📘 指南类（Guide）
- benchmark推理指南.md
- verl验证指南.md
- 配置文件指南.md

#### 📗 说明类（Explained）
- reward奖励计算说明.md
- 验证流程说明.md
- 多数投票说明.md

#### 📙 实现类（Implementation）
- 轨迹元数据实现.md

---

## 🔗 相关资源

### 官方文档
- VERL项目: https://github.com/volcengine/verl
- 主README: [../../README.md](../../README.md)
- CLAUDE说明: [../../CLAUDE.md](../../CLAUDE.md)

### 脚本文件
- 预处理脚本: `examples/data_preprocess/`
- 推理脚本: `examples/sglang_multiturn/search_browser/`
- 可视化脚本: `scripts/visualize_trajectory.py`

### 配置文件
- 基础配置: `verl/trainer/config/`
- 项目配置: `examples/sglang_multiturn/config/`

---

## ✏️ 文档维护

- 文档创建时间: 2024-10-26 至 2024-11-05
- 最后更新: 2024-11-05
- 维护者: Claude Code
- 语言: 中文

## 💡 使用建议

1. **新手**: 先看"快速开始"部分的文档
2. **调试**: 根据问题类型查找对应的"深入理解"文档
3. **高级功能**: 参考"高级功能"文档
4. **贡献**: 遵循项目的CONTRIBUTING.md规范

---

**注意**: 这些文档是在实际使用和开发过程中创建的，包含了大量实践经验和调试技巧。建议结合实际操作理解。
