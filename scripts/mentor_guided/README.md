# Mentor-Guided Adaptive Inference Framework

## 导师引导的自适应推理框架

### 核心思想

大模型不再机械地写八股文（如生成完整大纲），而是像导师一样，只负责攻克最难的"起步阶段"。通过监测小模型的"熵"指标，验证大模型输出对小模型的实际帮助程度，实现真正的高效交接。

### 工作原理

1. **大模型推理**：Streaming方式输出token
2. **小模型监测**：这些token连续输入小模型，计算小模型对下一个token的预测分布熵
3. **自适应切换**：当熵降低到阈值以下（表示小模型已有足够信心），切换到小模型独立推理

### 熵作为"有益性"指标

- **高熵**：小模型不确定，需要更多帮助
- **低熵**：小模型有信心，可以独立继续
- **熵减**：相比baseline的熵降低，表示大模型的帮助有效

## 文件说明

### mentor_guided_inference.py
核心实现文件，包含：
- `MentorGuidedInference`: 主要推理类
- `EntropyMetrics`: 熵相关指标
- `InferenceState`: 推理状态追踪

### test_multi_length.py
**主要测试脚本**：测试大模型提供不同长度token后，小模型接续推理的效果。

```bash
# 基本用法
python test_multi_length.py

# 自定义参数
python test_multi_length.py \
    --mentor-model Qwen/Qwen2.5-7B-Instruct \
    --student-model Qwen/Qwen2.5-1.5B-Instruct \
    --lengths 0,10,20,50,100,200 \
    --student-max-tokens 300

# 自定义prompt
python test_multi_length.py --prompt "你的问题..."
```

### analyze_entropy.py
熵轨迹分析和可视化脚本。

### test_mentor_guided.py
在多个测试问题上评估效果。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mentor-model` | Qwen/Qwen2.5-1.5B-Instruct | 大模型（导师） |
| `--student-model` | Qwen/Qwen2.5-0.5B-Instruct | 小模型（学生） |
| `--lengths` | 0,10,20,50,100,200 | 测试的大模型token长度列表 |
| `--student-max-tokens` | 200 | 小模型最大生成长度 |
| `--entropy-threshold` | 2.0 | 熵阈值（低于此值切换到小模型） |
| `--reduction-threshold` | 0.3 | 熵减阈值（30%减少视为有效） |
| `--temperature` | 0.7 | 采样温度 |

## 输出示例

```
Testing mentor lengths: [0, 10, 20, 50, 100, 200]
Length   0: initial_entropy=4.2341
Length  10: initial_entropy=3.8567
Length  20: initial_entropy=3.1234
Length  50: initial_entropy=2.4521
Length 100: initial_entropy=1.8234
Length 200: initial_entropy=1.5632

RESULTS SUMMARY
Baseline entropy (no mentor help): 4.2341
Best length: 100 tokens
Min entropy achieved: 1.5632

Entropy by mentor length:
    0 tokens: entropy=4.2341, reduction=+0.0%
   10 tokens: entropy=3.8567, reduction=+8.9%
   20 tokens: entropy=3.1234, reduction=+26.2%
   50 tokens: entropy=2.4521, reduction=+42.1%
  100 tokens: entropy=1.8234, reduction=+56.9%
  200 tokens: entropy=1.5632, reduction=+63.1%
```

## 后续集成

如果测试效果良好，可以将此方法集成到 Ensemble-Hub 主框架中：
1. 在 `ensemblehub/utils/` 添加熵计算工具
2. 在 `ensemblehub/ensemble_methods/output_aggregation/sentence_level/` 添加 `mentor_guided_selector.py`
3. 在 `ensemble.py` 的 `OUTPUT_AGGREGATORS` 注册新方法
