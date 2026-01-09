# Battery Agent 实验工作流程 (Workflow)

本文档详细说明如何运行完整的14天电池套利实验，以及如何生成所有实验结果和图表。

---

## 📁 项目结构概览

```
Battery_agent/
├── main.py                 # 主实验入口 (RuleAgent, SimpleLLM, CoT, Reflexion)
├── main_aga.py             # AGA/MetaReflexion 训练脚本
├── scripts/
│   ├── regenerate_results.py   # 重新生成基线结果 (Rule, Q-Learning, DQN, Simple LLM)
│   └── generate_figures.py     # 统一图表生成脚本
├── outputs/
│   ├── 14days_results/         # 14天实验结果与图表
│   │   ├── experiment_summary.json  # 结构化实验摘要
│   │   ├── profit_comparison.png
│   │   ├── action_distribution.png
│   │   ├── meta_reflexion_analysis.png
│   │   ├── llm_cost_efficiency.png
│   │   └── cot_ablation_study.png
│   ├── aga_training_results.json    # MetaReflexion训练结果
│   ├── cot_rl_experiment_results.json # CoT消融实验结果
│   └── _regen_cache_*.json          # 基线缓存数据
├── src/
│   ├── agents.py           # 所有Agent实现
│   ├── env.py              # 电池环境
│   └── rl_baselines.py     # RL基线 (Q-Learning, DQN, MPC)
└── data/
    └── caiso_enhanced_data.csv  # CAISO市场数据
```

---

## 🚀 快速开始：运行全部实验

### 一键运行所有实验并生成图表

```bash
# 1. 激活虚拟环境
source .venv/bin/activate

# 2. 运行完整实验流程
./run_all_experiments.sh
```

或者按步骤手动运行：

---

## 📊 分步运行指南

### Step 1: 运行基线实验 (Rule-Based, Q-Learning, DQN, Simple LLM)

```bash
python scripts/regenerate_results.py --methods rule-based q-learning dqn simple_llm --days 14
```

**输出文件:**
- `outputs/_regen_cache_rule-based.json`
- `outputs/_regen_cache_q-learning.json`
- `outputs/_regen_cache_dqn.json`
- `outputs/_regen_cache_simple_llm.json`
- `outputs/full_experiment_results_14days.csv`

**预计耗时:** ~5分钟 (RL训练) + ~10分钟 (Simple LLM 336次API调用)

---

### Step 2: 运行 MetaReflexion (AGA) 训练

```bash
python main_aga.py --days 14 --verbose
```

**输出文件:**
- `outputs/aga_training_results.json` - 训练过程数据
- `outputs/best_strategy_*.py` - 最佳生成策略代码

**预计耗时:** ~3分钟 (约15次LLM调用)

**关键指标:**
- 总利润: ~$42.35
- LLM调用: 15次
- 生成策略数: 14个
- Pass@1: 100%

---

### Step 3: 运行 CoT 消融实验

```bash
python -c "
from src.experiment import run_cot_ablation_study
results = run_cot_ablation_study(num_days=14)
import json
with open('outputs/cot_rl_experiment_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('CoT ablation study completed!')
"
```

**输出文件:**
- `outputs/cot_rl_experiment_results.json`

**消融变体:**
| 模型 | 描述 | 预期利润 |
|------|------|----------|
| Model A (Full) | 完整CoT + 推理奖励 | ~$17.32 |
| Model B (No Reward) | CoT但无推理奖励 | ~$12.44 |
| Model C (No CoT) | 无CoT直接输出 | ~-$3.94 |

**预计耗时:** ~30分钟 (约1000次LLM调用)

---

### Step 4: 生成所有实验图表

```bash
python scripts/generate_figures.py
```

**输出文件 (位于 `outputs/14days_results/`):**

| 文件名 | 内容描述 |
|--------|----------|
| `profit_comparison.png` | 14天总利润排名 + 累积利润曲线 |
| `action_distribution.png` | 各方法Action分布对比 (横向堆叠柱状图) |
| `meta_reflexion_analysis.png` | AGA分析: Pass@1 + 代码演进 + 阶梯曲线 |
| `llm_cost_efficiency.png` | LLM成本效率对比 |
| `cot_ablation_study.png` | CoT消融实验 + 阈值学习曲线 |

---

## 📈 实验结果摘要 (14天)

### 利润排名

| 排名 | 方法 | 总利润 | LLM调用 | 效率 ($/call) |
|:----:|------|-------:|--------:|--------------:|
| 🥇 | MetaReflexion (AGA) | $42.35 | 15 | $2.82 |
| 🥈 | Rule-Based | $34.23 | 0 | - |
| 🥉 | Q-Learning | $31.50 | 0 | - |
| 4 | CoT (Full) | $17.32 | 350 | $0.05 |
| 5 | DQN | $13.19 | 0 | - |
| 6 | CoT (No Reward) | $12.44 | 350 | $0.04 |
| 7 | Simple LLM | $5.36 | 336 | $0.02 |

### 关键发现

1. **MetaReflexion (AGA)** 以最少的LLM调用实现最高利润
   - 生成可执行Python代码 (Pass@1=100%)
   - 阶梯式学习曲线：每发现新逻辑利润跳升

2. **CoT是LLM Agent的关键**
   - 无CoT (Model C): 负利润 (-$3.94)
   - 有CoT (Model A): 正利润 ($17.32)
   - 提升: **5.4x**

3. **Simple LLM 的"盲目"行为**
   - 98% HOLD动作
   - 无法进行有意义的交易决策

---

## 🔧 配置说明

### 环境配置 (`configs/default.yaml`)

```yaml
battery:
  capacity_kwh: 13.5
  max_charge_kw: 5.0
  max_discharge_kw: 5.0
  roundtrip_efficiency: 0.9
  min_soc: 0.1
  max_soc: 0.95

experiment:
  days: 14
  initial_soc: 0.5
  seed: 42
```

### API配置 (`.env`)

```bash
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4
```

---

## 🧹 清理与重置

```bash
# 清理所有缓存和输出
rm -rf outputs/_regen_cache_*.json
rm -rf outputs/14days_results/*.png
rm -rf outputs/*.json

# 重新运行全部实验
python scripts/regenerate_results.py --force
python main_aga.py --days 14
python scripts/generate_figures.py
```

---

## 📝 完整运行脚本

创建 `run_all_experiments.sh`:

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "Battery Agent - Full Experiment Pipeline"
echo "=========================================="

# Step 1: 基线实验
echo -e "\n[1/4] Running baseline experiments..."
python scripts/regenerate_results.py --methods rule-based q-learning dqn simple_llm --days 14

# Step 2: MetaReflexion
echo -e "\n[2/4] Running MetaReflexion (AGA)..."
python main_aga.py --days 14

# Step 3: CoT消融实验 (可选，耗时较长)
# echo -e "\n[3/4] Running CoT ablation study..."
# python -c "from src.experiment import run_cot_ablation_study; ..."

# Step 4: 生成图表
echo -e "\n[4/4] Generating figures..."
python scripts/generate_figures.py

echo -e "\n=========================================="
echo "All experiments completed!"
echo "Results: outputs/14days_results/"
echo "=========================================="
```

---

## 📚 参考文献

- AGENT² Paper: Agent-Generates-Agent architecture
- Reflexion: Language Agents with Verbal Reinforcement Learning
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

---

*最后更新: 2026-01-09*
