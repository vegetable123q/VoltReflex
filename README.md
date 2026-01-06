# 🔋 LLM-Based Battery Arbitrage Agent with Reflexion

基于 LangGraph 的智能体系统，模拟家庭储能电池在分时电价市场中的套利行为。

## 📋 项目概览

本项目实现了一个**带反思机制 (Reflexion)** 的 AI Agent，能够：
- 在低电价时段充电储能
- 在高电价时段放电套利
- 每日总结交易得失，优化策略

### 核心假设
> 带有"反思机制"的 Agent 能通过总结每日的交易得失，优化第二天的策略，其收益将优于无记忆的 Zero-shot Agent 和基于规则的基线。

## 🏗️ 项目结构

```
Battery_agent/
├── data/
│   ├── generate_data.py      # 合成数据生成脚本
│   └── market_data.csv       # 市场数据 (timestamp, price, load)
├── src/
│   ├── __init__.py
│   ├── env.py                # BatteryEnv 类定义
│   ├── agents.py             # BaseAgent, RuleAgent, ReflexionAgent 定义
│   ├── prompts.py            # 存储所有的 System Prompts
│   └── utils.py              # 数据加载与绘图工具
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

创建 `.env` 文件并添加 OpenAI API Key：

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

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

### 3. ReflexionAgent (核心)
带反思机制的 Agent，使用 LangGraph 管理状态：
- **短期记忆**: 记录当天的交易
- **长期记忆**: 存储每日反思总结
- **每日反思**: 分析错误，更新策略

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

## 📊 输出结果

运行后会生成：
- `experiment_results.csv` - 实验结果数据
- `cumulative_profits.png` - 累积利润对比图
- `daily_profits.png` - 每日利润对比图
- `action_distribution.png` - 动作分布图
- `soc_profile_*.png` - SOC 变化曲线

## 🔧 技术栈

- **Python 3.10+**
- **LangChain** - LLM 编排框架
- **LangGraph** - Agent 状态图管理
- **OpenAI API** - GPT-4o-mini / GPT-4o
- **Pandas** - 数据处理
- **Matplotlib/Seaborn** - 可视化

## 📈 预期结果

在典型的 7 天模拟中：
- **RuleAgent**: 稳定但保守的收益
- **SimpleLLMAgent**: 波动较大，可能出现错误决策
- **ReflexionAgent**: 随着反思积累，后期表现逐渐优于前两者

## 🧪 扩展实验

1. **调整反思频率**: 每小时/每周反思
2. **增加市场波动**: 在数据中添加随机事件
3. **多电池协同**: 扩展到多个电池系统
4. **真实数据**: 接入 CAISO 等真实电价数据

## 📄 License

MIT License
