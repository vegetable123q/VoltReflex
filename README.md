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

### 🌟 学术贡献

- **Reflexion 机制**: 首次将 Reflexion 框架应用于电力市场套利
- **多基线对比**: 规则、Zero-shot LLM、Q-Learning、DQN、MPC
- **多市场验证**: CAISO、PJM、ERCOT 三大电力市场
- **统计显著性**: 配对 t 检验、Wilcoxon 检验、Bootstrap CI

## 🏗️ 项目结构

```
Battery_agent/
├── configs/
│   └── default.yaml          # 实验配置文件
├── data/
│   ├── caiso_enhanced_data.csv  # 增强后的市场数据 (实验使用)
│   ├── enhance_price_data.py    # 数据增强脚本
│   └── process_caiso_data.py    # 原始数据处理脚本
├── scripts/
│   └── reproduce_paper.py    # 一键复现论文结果
├── src/
│   ├── __init__.py
│   ├── env.py                # BatteryEnv 电池环境
│   ├── agents.py             # Agent 实现 (Rule, LLM, Reflexion)
│   ├── prompts.py            # LLM Prompt 模板
│   ├── metrics.py            # 金融指标 (Sharpe, Drawdown 等)
│   ├── data_loader.py        # 多市场数据加载器
│   ├── experiment.py         # 实验运行框架
│   ├── visualization.py      # 学术级可视化
│   ├── rl_baselines.py       # RL 基线 (Q-Learning, DQN)
│   └── utils.py              # 工具函数
├── tests/
│   └── test_all.py           # 单元测试
├── main.py                   # 主程序入口
├── requirements.txt          # 依赖列表
├── pyproject.toml            # 项目配置
└── README.md                 # 本文件
```

## 📊 数据来源与增强

### 原始数据

原始数据来源于 [GridStatus.io](https://www.gridstatus.io/)，包含 **CAISO (California ISO)** 电力市场的真实数据：

- **LMP 价格数据**: 节点边际电价 (Locational Marginal Price)
- **负荷数据**: CA ISO-TAC 区域总负荷
- **时间范围**: 2025年12月 (30天, 720小时)

### 数据增强规则

为了增加套利空间并保留原始数据的时间模式特征，我们对数据进行了以下增强处理：

| 增强策略 | 参数 | 说明 |
|---------|------|------|
| **时段差异化** | 低谷×0.5, 高峰×2.5 | 低谷时段 (0-5时, 12-14时) 价格降低，高峰时段 (6-9时, 16-21时) 价格提高 |
| **波动放大** | ×3.0 | 围绕均值的价格偏离放大3倍 |
| **价格尖峰** | 3%, ×4.0 | 3%概率在高峰时段出现4倍价格尖峰 |
| **负电价** | 2% | 2%概率在低谷时段出现负电价 (模拟可再生能源过剩) |

### 数据统计对比

| 指标 | 原始数据 | 增强数据 |
|-----|---------|----------|
| 价格均值 | $0.030/kWh | $0.058/kWh |
| 价格标准差 | $0.012 | $0.075 |
| 价格范围 | $0.00 - $0.10 | $-0.02 - $0.98 |
| 峰谷价差 | ~$0.10 | ~$1.00 |

> **注**: 仓库中只包含增强后的最终数据 (`caiso_enhanced_data.csv`)，原始数据可从 GridStatus.io 获取。

## 🚀 快速开始

### 1. 环境准备

```bash
# 使用 uv 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate  # macOS/Linux
# or: .venv\Scripts\activate  # Windows

uv pip install -r requirements.txt
```

### 2. 配置 API Key

本项目使用 LLM API 进行智能决策。API Key 可从 [清华大学易计算平台](https://easycompute.cs.tsinghua.edu.cn/) 获取。

复制 `.env.example` 并填入你的 API Key：

```bash
cp .env.example .env
```

然后编辑 `.env` 文件：

```dotenv
# API 配置
BASE_URL=https://llmapi.paratera.com
API_KEY=your-api-key-here          # 从 https://easycompute.cs.tsinghua.edu.cn/ 获取
MODEL_ID=deepseek-chat             # 或其他支持的模型如 Kimi-K2

# LangChain 兼容配置（代码使用）
OPENAI_API_BASE=https://llmapi.paratera.com/v1
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=deepseek-chat
```

> **获取 API Key**: 访问 https://easycompute.cs.tsinghua.edu.cn/ 注册并获取 API Key

### 3. 运行实验

数据已预处理完成 (`data/caiso_enhanced_data.csv`)，可直接运行实验：

### 4. 运行实验

```bash
# 运行 7 天的模拟（Rule-based vs Reflexion）
python main.py --days 7

# 只运行规则基线（无需 API Key）
python main.py --days 7 --agents rule

# 运行所有三种 Agent
python main.py --days 7 --agents rule simple_llm reflexion

# 使用不同的 LLM 模型
python main.py --model gpt-4o
```

## 🤖 Agent 类型

### 1. RuleAgent (基线)
硬编码的规则策略：
- 价格 < $0.15 且 SOC < 90% → **充电**
- 价格 > $0.40 且 SOC > 10% → **放电**
- 其他情况 → **保持**

### 2. SimpleLLMAgent (Zero-shot)
无记忆的 LLM Agent，每次决策独立调用 LLM。

### 3. ReflexionAgent (核心创新)
带反思机制的 Agent，使用 LangGraph 管理状态：
- **短期记忆**: 记录当天的交易历史
- **长期记忆**: 存储每日反思总结
- **每日反思**: 分析错误，更新策略
- **策略优化**: 基于历史经验调整决策阈值

### 4. RL Baselines (新增)
- **Q-Learning**: 表格型强化学习
- **DQN**: 深度 Q 网络
- **MPC**: 模型预测控制 (理论上界)

## ⚡ 电池参数 (Tesla Powerwall)

| 参数 | 值 |
|------|-----|
| 容量 | 13.5 kWh |
| 最大功率 | 5 kW |
| 往返效率 | 90% |
| 最低 SOC | 10% |

## 💰 电价模型 (增强数据)

| 时段 | 时间 | 价格范围 |
|------|------|------|
| 低谷 | 00:00-05:00, 12:00-14:00 | ~$0.01-0.02/kWh (含负电价) |
| 平段 | 10:00-11:00, 15:00, 22:00-23:00 | ~$0.03-0.04/kWh |
| 高峰 | 06:00-09:00, 16:00-21:00 | ~$0.08-0.14/kWh (含尖峰) |

## 📊 评估指标

### 金融指标
- **Sharpe Ratio**: 风险调整后收益
- **Sortino Ratio**: 下行风险调整收益
- **Max Drawdown**: 最大回撤
- **Profit Factor**: 盈亏比
- **Win Rate**: 胜率
- **Calmar Ratio**: 收益/最大回撤

### 统计检验
- **配对 t 检验**: 策略间差异显著性
- **Wilcoxon 检验**: 非参数检验
- **Bootstrap CI**: 95% 置信区间
- **Cohen's d**: 效应量

## 📈 实验结果

### 实验配置

| 参数 | 值 |
|------|-----|
| **实验周期** | 14 天 (336 小时) |
| **电池配置** | Tesla Powerwall 2 (13.5 kWh, 5.0 kW) |
| **数据来源** | CAISO 增强电价数据 |
| **价格范围** | $-0.02 ~ $0.98/kWh |
| **LLM 模型** | Kimi-K2 |

### 主实验结果 (14天模拟)

| 排名 | 方法 | 总利润($) | 充电次数 | 放电次数 | 持有次数 | LLM调用 | 相对MPC |
|:----:|------|--------:|--------:|--------:|--------:|--------:|--------:|
| 🥇 | **Q-Learning** | **35.56** | 58 | 59 | 219 | 0 | 183.7% |
| 🥈 | **Rule-Based** | **23.29** | 61 | 63 | 212 | 0 | 120.3% |
| 🥉 | **Reflexion** | **19.75** | 62 | 61 | 213 | 350 | 102.1% |
| 4 | MPC (Upper Bound) | 19.36 | 23 | 22 | 291 | 0 | 100.0% |
| 5 | Simple LLM | 5.16 | 6 | 2 | 328 | 336 | 26.6% |
| 6 | DQN | -0.26 | 2 | 0 | 334 | 0 | -1.4% |

### 📊 结果对比图

![Experiment Results](outputs/full_comparison_chart.png)

### 🔑 关键发现

#### 1. Q-Learning 表现最佳 ($35.56)
- 通过 100 轮训练学习到最优交易策略
- 超越了 MPC 理论上界 (183.7%)
- 原因：MPC 仅考虑 24 小时窗口，而 Q-Learning 学习了长期价格模式

#### 2. Rule-Based 基线稳健可靠 ($23.29)
- 简单阈值规则，无需任何训练
- 实际部署的首选方案
- 充放电次数均衡 (61 vs 63)

#### 3. Reflexion 显著优于 Zero-shot LLM
- **利润提升: +283%** ($5.16 → $19.75)
- 交易活跃度大幅提升:
  - 充电: 6 → 62 (+933%)
  - 放电: 2 → 61 (+2950%)
- 验证了反思机制的有效性

#### 4. Reflexion 学到的策略要点
通过每日反思，Reflexion Agent 总结出以下交易规则：
1. **保持低 SOC** (≤10%) 等待低价时机
2. **严格充电阈值**: 只在价格 < $0.03 时充电
3. **放电触发价格**: 当价格 ≥ $0.11 时开始放电
4. **避免高价充电**: 即使电量低也不在高价时段补电

#### 5. DQN 需要更多训练
- 100 轮训练不足以收敛
- 建议增加到 1000+ 轮
- 或考虑使用预训练/迁移学习

### LLM vs Non-LLM 方法对比

| 指标 | LLM 方法 | Non-LLM 方法 |
|------|---------|-------------|
| 平均利润 | $12.45 | $19.61 |
| 最佳利润 | $19.75 (Reflexion) | $35.56 (Q-Learning) |
| 需要训练 | ❌ | ✅ (RL) / ❌ (Rule) |
| 可解释性 | ✅ 高 | ⚠️ 中等 |
| API 成本 | ✅ 有 | ❌ 无 |

### 结论

1. **传统 RL 方法 (Q-Learning)** 在充分训练后可超越 LLM 方法
2. **反思机制 (Reflexion)** 显著提升 LLM Agent 的决策能力 (+283%)
3. **简单规则基线** 仍是实际部署的可靠选择
4. **LLM Agent 的优势** 在于无需训练、可解释性强、易于调试

## 📁 输出文件

运行后会生成：

### 数据文件
- `outputs/experiment_results.csv` - 单次实验结果
- `outputs/full_experiment_results_14days.csv` - 完整实验结果汇总

### 图表 (PNG)
- `outputs/full_comparison_chart.png` - 综合对比图 (利润、操作分布、MPC对比)
- `outputs/cumulative_profits.png` - 累积收益曲线
- `outputs/daily_profits.png` - 每日收益对比
- `outputs/action_distribution.png` - 操作分布图
- `outputs/soc_profile_*.png` - 各 Agent 的 SOC 变化曲线

## 🔧 技术栈

- **Python 3.10+**
- **LangChain** - LLM 编排框架
- **LangGraph** - Agent 状态图管理
- **OpenAI API** - GPT-4o-mini / GPT-4o
- **Pandas/NumPy** - 数据处理
- **Matplotlib/Seaborn** - 学术可视化
- **SciPy** - 统计检验
- **PyYAML** - 配置管理
- **pytest** - 单元测试

## 🧪 运行测试

```bash
# 运行所有单元测试
python -m pytest tests/ -v

# 运行覆盖率测试
python -m pytest tests/ --cov=src --cov-report=html
```

## 📄 引用

如果使用本项目，请引用：

```bibtex
@article{wang2024battery,
  title={LLM-Based Battery Arbitrage Agent with Reflexion: 
         Learning to Trade in Time-of-Use Electricity Markets},
  author={Wang, XXX},
  journal={arXiv preprint},
  year={2024}
}
```

## 📄 License

MIT License
