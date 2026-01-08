# 🔋 LLM-Based Battery Arbitrage Agent with Reflexion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

实验设置与运行方式请见 `startup.md`；本 README 仅展示实验结果。

## 📈 实验结果

### 主实验结果（14 天）

<!-- RESULTS_TABLE_START -->
| 排名 | 方法 | 总利润($) | 充电 | 放电 | 持有 | LLM调用 | 相对MPC |
|:----:|------|--------:|-----:|-----:|-----:|--------:|--------:|
| 🥇 | Rule-Based | 34.23 | 63 | 64 | 209 | 0 | 184.1% |
| 🥈 | Q-Learning | 31.50 | 57 | 57 | 222 | 0 | 169.4% |
| 🥉 | MPC (24h) | 18.59 | 24 | 21 | 291 | 0 | 100.0% |
| 4 | Reflexion | 13.34 | 29 | 22 | 285 | 350 | 71.7% |
| 5 | DQN | 13.19 | 43 | 42 | 251 | 0 | 70.9% |
| 6 | Simple LLM | 5.36 | 6 | 2 | 328 | 336 | 28.8% |
<!-- RESULTS_TABLE_END -->

完整结果：`outputs/full_experiment_results_14days.csv`（相对MPC以 `MPC (24h)` 为 100%）。

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

## 📄 License

MIT License
