# 🚀 快速启动指南

本文档包含项目的环境配置、运行说明和技术细节。

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
├── outputs/                  # 实验输出目录
├── main.py                   # 主程序入口
├── requirements.txt          # 依赖列表
├── pyproject.toml            # 项目配置
├── README.md                 # 项目介绍与实验结果
└── startup.md                # 本文件
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

## 🛠️ 环境准备

### 1. 安装依赖

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

## 🚀 运行实验

数据已预处理完成 (`data/caiso_enhanced_data.csv`)，可直接运行实验：

```bash
# 运行 14 天的模拟（与配置文件一致）
python main.py --days 14

# 只运行规则基线（无需 API Key）
python main.py --days 14 --agents rule

# 运行单个 LLM Agent
python main.py --days 14 --agents simple_llm
python main.py --days 14 --agents reflexion

# 运行所有三种 Agent
python main.py --days 14 --agents rule simple_llm reflexion

# 使用不同的 LLM 模型
python main.py --model gpt-4o
```

### 运行 RL 基线

```python
# 在 Python 中运行 RL 基线对比
from src.rl_baselines import compare_rl_baselines
from src.utils import load_market_data

df = load_market_data()
df = df.head(14 * 24)  # 14天数据
results = compare_rl_baselines(df, n_episodes=100)
```

## 🤖 Agent 类型详解

### 1. RuleAgent (基线)
硬编码的规则策略：
- 价格 < 阈值 且 SOC < 90% → **充电**
- 价格 > 阈值 且 SOC > 10% → **放电**
- 其他情况 → **保持**

### 2. SimpleLLMAgent (Zero-shot)
无记忆的 LLM Agent，每次决策独立调用 LLM。

### 3. ReflexionAgent (核心创新)
带反思机制的 Agent，使用 LangGraph 管理状态：
- **短期记忆**: 记录当天的交易历史
- **长期记忆**: 存储每日反思总结
- **每日反思**: 分析错误，更新策略
- **策略优化**: 基于历史经验调整决策阈值

### 4. RL Baselines
- **Q-Learning**: 表格型强化学习 (100轮训练)
- **DQN**: 深度 Q 网络 (100轮训练)
- **MPC**: 模型预测控制 (24小时窗口，理论上界参考)

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

## 📁 输出文件

运行后会生成：

### 数据文件
- `outputs/experiment_results.csv` - 单次实验结果
- `outputs/full_experiment_results_14days.csv` - 完整实验结果汇总

### 图表 (PNG)
- `outputs/full_comparison_chart.png` - 综合对比图
- `outputs/cumulative_profits.png` - 累积收益曲线
- `outputs/daily_profits.png` - 每日收益对比
- `outputs/action_distribution.png` - 操作分布图
- `outputs/soc_profile_*.png` - 各 Agent 的 SOC 变化曲线

## 🔧 技术栈

- **Python 3.10+**
- **LangChain** - LLM 编排框架
- **LangGraph** - Agent 状态图管理
- **OpenAI API** - GPT-4o-mini / GPT-4o / Kimi-K2
- **Pandas/NumPy** - 数据处理
- **Matplotlib/Seaborn** - 学术可视化
- **PyYAML** - 配置管理
- **pytest** - 单元测试

## 🧪 运行测试

```bash
# 运行所有单元测试
python -m pytest tests/ -v

# 运行覆盖率测试
python -m pytest tests/ --cov=src --cov-report=html
```
