"""
Utilities Module
数据加载和可视化工具
"""
import os
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_market_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    加载市场数据
    
    Args:
        file_path: CSV 文件路径。如果为 None，使用默认路径
        
    Returns:
        包含 timestamp, price, load 的 DataFrame
    """
    if file_path is None:
        # 默认路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, '..', 'data', 'market_data.csv')
    
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def plot_price_profile(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    绘制电价曲线
    
    Args:
        df: 市场数据 DataFrame
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(df['timestamp'], df['price'], 'b-', alpha=0.7, linewidth=1)
    ax.fill_between(df['timestamp'], df['price'], alpha=0.3)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Price ($/kWh)')
    ax.set_title('Electricity Price Profile')
    
    # 格式化 x 轴
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 图表已保存: {save_path}")
    
    plt.show()


def plot_cumulative_profits(
    results: Dict[str, List[float]],
    title: str = "Cumulative Profit Comparison",
    save_path: Optional[str] = None
):
    """
    绘制累积利润对比曲线
    
    Args:
        results: {agent_name: [hourly_profits]} 字典
        title: 图表标题
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    for idx, (agent_name, profits) in enumerate(results.items()):
        cumulative = []
        total = 0
        for p in profits:
            total += p
            cumulative.append(total)
        
        color = colors[idx % len(colors)]
        ax.plot(cumulative, label=f'{agent_name} (Total: ${total:.2f})', 
                linewidth=2, color=color)
    
    ax.set_xlabel('Hour')
    ax.set_ylabel('Cumulative Profit ($)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 图表已保存: {save_path}")
    
    plt.show()


def plot_daily_profits(
    results: Dict[str, List[float]],
    title: str = "Daily Profit Comparison",
    save_path: Optional[str] = None
):
    """
    绘制每日利润柱状图对比
    
    Args:
        results: {agent_name: [daily_profits]} 字典
        title: 图表标题
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n_agents = len(results)
    n_days = len(list(results.values())[0])
    
    x = range(n_days)
    width = 0.8 / n_agents
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    for idx, (agent_name, daily_profits) in enumerate(results.items()):
        offset = (idx - n_agents/2 + 0.5) * width
        bars = ax.bar([i + offset for i in x], daily_profits, 
                      width=width, label=agent_name, color=colors[idx % len(colors)],
                      alpha=0.8)
    
    ax.set_xlabel('Day')
    ax.set_ylabel('Daily Profit ($)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Day {i+1}' for i in x], rotation=45)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 图表已保存: {save_path}")
    
    plt.show()


def plot_soc_profile(
    history: List[Dict],
    price_data: pd.DataFrame,
    title: str = "Battery SOC and Price Profile",
    save_path: Optional[str] = None
):
    """
    绘制电池 SOC 和电价的双轴图
    
    Args:
        history: 环境的历史记录
        price_data: 价格数据
        title: 图表标题
        save_path: 保存路径
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # SOC 曲线
    steps = [h['step'] for h in history]
    socs = [h['soc_after'] for h in history]
    
    ax1.plot(steps, socs, 'g-', linewidth=2, label='Battery SOC (%)')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('SOC (%)', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.set_ylim(0, 100)
    
    # 价格曲线（双轴）
    ax2 = ax1.twinx()
    prices = price_data['price'].values[:len(steps)]
    ax2.plot(steps, prices, 'b--', alpha=0.6, linewidth=1, label='Price ($/kWh)')
    ax2.set_ylabel('Price ($/kWh)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    # 标记充放电动作
    for h in history:
        step = h['step']
        if h['action'] == 'CHARGE':
            ax1.axvline(x=step, color='green', alpha=0.1)
        elif h['action'] == 'DISCHARGE':
            ax1.axvline(x=step, color='red', alpha=0.1)
    
    ax1.set_title(title)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 图表已保存: {save_path}")
    
    plt.show()


def plot_action_distribution(
    results: Dict[str, Dict[str, int]],
    title: str = "Action Distribution by Agent",
    save_path: Optional[str] = None
):
    """
    绘制各 Agent 的动作分布饼图
    
    Args:
        results: {agent_name: {action: count}} 字典
        title: 图表标题
        save_path: 保存路径
    """
    n_agents = len(results)
    fig, axes = plt.subplots(1, n_agents, figsize=(4*n_agents, 4))
    
    if n_agents == 1:
        axes = [axes]
    
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # CHARGE, DISCHARGE, HOLD
    
    for ax, (agent_name, actions) in zip(axes, results.items()):
        counts = [actions.get('CHARGE', 0), 
                  actions.get('DISCHARGE', 0), 
                  actions.get('HOLD', 0)]
        labels = ['CHARGE', 'DISCHARGE', 'HOLD']
        
        ax.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90)
        ax.set_title(agent_name)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 图表已保存: {save_path}")
    
    plt.show()


def print_experiment_summary(results: Dict[str, Dict]):
    """
    打印实验结果摘要
    
    Args:
        results: {agent_name: {metrics}} 字典
    """
    print("\n" + "="*60)
    print("📊 EXPERIMENT SUMMARY")
    print("="*60)
    
    for agent_name, metrics in results.items():
        print(f"\n🤖 {agent_name}:")
        print(f"   Total Profit: ${metrics.get('total_profit', 0):.4f}")
        print(f"   Total Cost:   ${metrics.get('total_cost', 0):.4f}")
        print(f"   Total Revenue: ${metrics.get('total_revenue', 0):.4f}")
        print(f"   LLM Calls:    {metrics.get('llm_calls', 0)}")
        print(f"   Actions: CHARGE={metrics.get('charge_count', 0)}, "
              f"DISCHARGE={metrics.get('discharge_count', 0)}, "
              f"HOLD={metrics.get('hold_count', 0)}")
    
    print("\n" + "="*60)
    
    # 找出最佳 Agent
    best_agent = max(results.items(), key=lambda x: x[1].get('total_profit', 0))
    print(f"🏆 Best Performer: {best_agent[0]} with ${best_agent[1]['total_profit']:.4f} profit")
    print("="*60 + "\n")


def create_results_dataframe(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    将结果转换为 DataFrame
    
    Args:
        results: {agent_name: {metrics}} 字典
        
    Returns:
        结果 DataFrame
    """
    rows = []
    for agent_name, metrics in results.items():
        rows.append({
            'Agent': agent_name,
            'Total Profit ($)': metrics.get('total_profit', 0),
            'Total Cost ($)': metrics.get('total_cost', 0),
            'Total Revenue ($)': metrics.get('total_revenue', 0),
            'LLM Calls': metrics.get('llm_calls', 0),
            'Charge Actions': metrics.get('charge_count', 0),
            'Discharge Actions': metrics.get('discharge_count', 0),
            'Hold Actions': metrics.get('hold_count', 0)
        })
    
    return pd.DataFrame(rows)
