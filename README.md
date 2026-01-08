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
│   ├── generate_data.py      # 合成数据生成脚本
│   └── market_data.csv       # 市场数据 (timestamp, price, load)
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

### 3. 生成市场数据

```bash
python data/generate_data.py
```

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

## 💰 电价模型

| 时段 | 时间 | 价格 |
|------|------|------|
| 低谷 | 23:00-07:00 | ~$0.10/kWh |
| 平段 | 07:00-17:00, 21:00-23:00 | ~$0.20/kWh |
| 高峰 | 17:00-21:00 | ~$0.50/kWh |

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

### 主实验 (14天模拟)

| 策略 | 总收益 | Sharpe | Max DD | Win Rate |
|------|--------|--------|--------|----------|
| Rule-based | $X.XX | X.XX | $X.XX | XX% |
| Zero-shot | $X.XX | X.XX | $X.XX | XX% |
| Q-Learning | $X.XX | X.XX | $X.XX | XX% |
| DQN | $X.XX | X.XX | $X.XX | XX% |
| **Reflexion** | **$X.XX** | **X.XX** | **$X.XX** | **XX%** |
| MPC (上界) | $X.XX | X.XX | $X.XX | XX% |

### 跨市场验证

| 市场 | Rule | Reflexion | 提升 |
|------|------|-----------|------|
| CAISO | $X.XX | $X.XX | +XX% |
| PJM | $X.XX | $X.XX | +XX% |
| ERCOT | $X.XX | $X.XX | +XX% |

## 🔬 消融实验

### 记忆窗口大小
| 窗口 (天) | 收益 | Sharpe |
|-----------|------|--------|
| 1 | $X.XX | X.XX |
| 3 | $X.XX | X.XX |
| 7 | $X.XX | X.XX |

### 反思频率
| 频率 | 收益 | LLM调用次数 |
|------|------|-------------|
| 每小时 | $X.XX | XXX |
| 每日 | $X.XX | XX |
| 每周 | $X.XX | X |

## 📁 输出文件

运行后会生成：

### 数据文件
- `experiment_results.csv` - 实验结果数据
- `all_results_summary.json` - 完整结果汇总

### 图表 (PNG/PDF/SVG)
- `fig1_cumulative_profits.pdf` - 累积收益曲线 (带置信区间)
- `fig2_daily_boxplot.pdf` - 每日收益箱线图
- `fig3_rl_training.pdf` - RL 训练曲线
- `fig4_cross_market.pdf` - 跨市场对比
- `fig5_ablation.pdf` - 消融实验热力图

### LaTeX 表格
- `table1_main_results.tex` - 主实验结果
- `table2_rl_comparison.tex` - RL 基线对比
- `table3_cross_market.tex` - 跨市场结果

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
