# 🔋 LLM-Based Battery Arbitrage Agent with Reflexion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

基于 LangGraph 的智能体系统，模拟家庭储能电池在分时电价市场中的套利行为。

## 📋 项目概览

本项目实现了一个**带反思机制 (Reflexion)** 的 AI Agent，能够：
- 在低电价时段充电储能
- 在高电价时段放电套利
- 每日总结交易得失，优化策略

### 核心假设
> 带有"反思机制"的 Agent 能通过总结每日的交易得失，优化第二天的策略，其收益将优于无记忆的 Zero-shot Agent 和基于规则的基线。

> 📖 **快速开始**: 详细的环境配置和运行说明请参考 [startup.md](startup.md)

---

## 📈 实验结果

### 实验配置

| 参数 | 值 |
|------|-----|
| **实验周期** | 14 天 (336 小时) |
| **电池配置** | Tesla Powerwall 2 (13.5 kWh, 5.0 kW) |
| **数据来源** | CAISO 增强电价数据 |
| **价格范围** | $-0.02 ~ $0.98/kWh |
| **LLM 模型** | Kimi-K2 |

### 主实验结果

| 排名 | 方法 | 总利润($) | 充电 | 放电 | 持有 | LLM调用 | 相对MPC |
|:----:|------|--------:|-----:|-----:|-----:|--------:|--------:|
| 🥇 | **Q-Learning** | **35.56** | 58 | 59 | 219 | 0 | 183.7% |
| 🥈 | **Rule-Based** | **23.29** | 61 | 63 | 212 | 0 | 120.3% |
| 🥉 | **Reflexion** | **19.75** | 62 | 61 | 213 | 350 | 102.1% |
| 4 | MPC (Upper Bound) | 19.36 | 23 | 22 | 291 | 0 | 100.0% |
| 5 | Simple LLM | 5.16 | 6 | 2 | 328 | 336 | 26.6% |
| 6 | DQN | -0.26 | 2 | 0 | 334 | 0 | -1.4% |

### 📊 结果可视化

<p align="center">
  <img src="outputs/full_comparison_chart.png" alt="Experiment Results" width="100%">
</p>

<details>
<summary>📈 更多可视化图表</summary>

#### 累积利润曲线
<img src="outputs/cumulative_profits.png" alt="Cumulative Profits" width="80%">

#### 每日利润对比
<img src="outputs/daily_profits.png" alt="Daily Profits" width="80%">

#### 操作分布
<img src="outputs/action_distribution.png" alt="Action Distribution" width="80%">

</details>

---

## 🔑 关键发现

### 1. Q-Learning 表现最佳 ($35.56)
- 通过 100 轮训练学习到最优交易策略
- 超越了 MPC 理论上界 (183.7%)
- 原因：MPC 仅考虑 24 小时窗口，Q-Learning 学习了长期价格模式

### 2. Rule-Based 基线稳健可靠 ($23.29)
- 简单阈值规则，无需任何训练
- 实际部署的首选方案

### 3. Reflexion 显著优于 Zero-shot LLM
- **利润提升: +283%** ($5.16 → $19.75)
- 交易活跃度大幅提升 (充电: 6→62, 放电: 2→61)
- 验证了反思机制的有效性

### 4. Reflexion 学到的策略
通过每日反思，Agent 总结出：
1. 保持低 SOC (≤10%) 等待低价
2. 只在价格 < $0.03 时充电
3. 当价格 ≥ $0.11 时放电

### 5. DQN 需要更多训练
- 100 轮训练不足以收敛
- 建议增加到 1000+ 轮

---

## 📊 LLM vs Non-LLM 对比

| 指标 | LLM 方法 | Non-LLM 方法 |
|------|---------|-------------|
| 平均利润 | $12.45 | $19.61 |
| 最佳利润 | $19.75 (Reflexion) | $35.56 (Q-Learning) |
| 需要训练 | ❌ | ✅ (RL) / ❌ (Rule) |
| 可解释性 | ✅ 高 | ⚠️ 中等 |
| API 成本 | ✅ 有 | ❌ 无 |

---

## 💡 结论

1. **传统 RL (Q-Learning)** 在充分训练后可超越 LLM 方法
2. **反思机制 (Reflexion)** 显著提升 LLM Agent 决策能力 (+283%)
3. **简单规则基线** 仍是实际部署的可靠选择
4. **LLM Agent 优势** 在于无需训练、可解释性强

---

## ⚠️ 实验状态

| 实验 | 状态 | 说明 |
|------|------|------|
| Rule-Based Agent | ✅ 完成 | 14天，$23.29 |
| Simple LLM Agent | ✅ 完成 | 14天，$5.16，336次LLM调用 |
| Reflexion Agent | ✅ 完成 | 14天，$19.75，350次LLM调用 |
| Q-Learning | ✅ 完成 | 100轮训练，$35.56 |
| DQN | ✅ 完成 | 100轮训练，$-0.26 (需更多训练) |
| MPC (Upper Bound) | ✅ 完成 | 24小时窗口，$19.36 |
| 多市场验证 (PJM/ERCOT) | ❌ 未做 | 需额外数据 |
| 消融实验 | ❌ 未做 | 记忆窗口、反思频率 |
| 统计显著性检验 | ❌ 未做 | t检验、Bootstrap CI |

---

## 📄 License

MIT License
