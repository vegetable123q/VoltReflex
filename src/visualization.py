"""
学术级可视化模块
生成适合论文发表的高质量图表
"""
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats

# 设置学术论文风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.figsize': (7, 4.5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 学术配色方案
COLORS = {
    'rule': '#1f77b4',      # 蓝色
    'simple_llm': '#ff7f0e', # 橙色
    'reflexion': '#2ca02c',  # 绿色
    'optimal': '#d62728',    # 红色
    'baseline': '#7f7f7f',   # 灰色
}

AGENT_LABELS = {
    'RuleAgent': 'Rule-based',
    'SimpleLLMAgent': 'Zero-shot LLM',
    'ReflexionAgent': 'Reflexion (Ours)',
}


class AcademicVisualizer:
    """
    学术级可视化器
    生成符合顶级会议/期刊标准的图表
    """
    
    def __init__(self, output_dir: str = "figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_cumulative_profits_with_ci(
        self,
        results: Dict[str, List],
        title: str = "",
        save_name: str = "cumulative_profit"
    ):
        """
        带置信区间的累积利润曲线
        适合展示多次运行的结果
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for agent_name, runs in results.items():
            # 聚合多次运行
            all_cumulative = []
            for run in runs:
                profits = run.get('hourly_profits', [])
                cumulative = np.cumsum(profits)
                all_cumulative.append(cumulative)
            
            if not all_cumulative:
                continue
            
            # 对齐长度
            min_len = min(len(c) for c in all_cumulative)
            all_cumulative = np.array([c[:min_len] for c in all_cumulative])
            
            mean = np.mean(all_cumulative, axis=0)
            std = np.std(all_cumulative, axis=0)
            
            # 95% 置信区间
            ci = 1.96 * std / np.sqrt(len(all_cumulative))
            
            hours = np.arange(len(mean))
            
            color = COLORS.get(agent_name.lower().replace('agent', ''), '#333333')
            label = AGENT_LABELS.get(agent_name, agent_name)
            
            ax.plot(hours, mean, label=label, color=color, linewidth=2)
            ax.fill_between(hours, mean - ci, mean + ci, color=color, alpha=0.2)
        
        ax.set_xlabel('Hour')
        ax.set_ylabel('Cumulative Profit ($)')
        ax.set_title(title or 'Cumulative Profit Comparison')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 添加日分隔线
        for day in range(1, len(mean) // 24 + 1):
            ax.axvline(x=day * 24, color='gray', linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(save_name)
        plt.show()
    
    def plot_daily_boxplot(
        self,
        results: Dict[str, List],
        save_name: str = "daily_boxplot"
    ):
        """
        每日利润箱线图
        展示收益分布
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 准备数据
        data = []
        for agent_name, runs in results.items():
            for run in runs:
                daily = run.get('daily_profits', [])
                for day, profit in enumerate(daily):
                    data.append({
                        'Agent': AGENT_LABELS.get(agent_name, agent_name),
                        'Day': day + 1,
                        'Profit': profit
                    })
        
        df = pd.DataFrame(data)
        
        # 绘制
        palette = {AGENT_LABELS.get(k, k): COLORS.get(k.lower().replace('agent', ''), '#333') 
                   for k in results.keys()}
        
        sns.boxplot(
            data=df,
            x='Day',
            y='Profit',
            hue='Agent',
            palette=palette,
            ax=ax
        )
        
        ax.set_xlabel('Day')
        ax.set_ylabel('Daily Profit ($)')
        ax.set_title('Daily Profit Distribution')
        ax.legend(title='', loc='upper right')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        self._save_figure(save_name)
        plt.show()
    
    def plot_action_heatmap(
        self,
        history: List[Dict],
        agent_name: str = "Agent",
        save_name: str = "action_heatmap"
    ):
        """
        动作-价格-时间热力图
        展示 Agent 的决策模式
        """
        # 构建矩阵: 行=天, 列=小时, 值=动作
        days = max(h.get('day', 0) for h in history) + 1
        hours = 24
        
        action_map = {'CHARGE': -1, 'HOLD': 0, 'DISCHARGE': 1}
        matrix = np.zeros((days, hours))
        price_matrix = np.zeros((days, hours))
        
        for h in history:
            day = h.get('day', 0)
            hour = h.get('hour', 0)
            action = h.get('action', 'HOLD')
            price = h.get('price', 0)
            
            if day < days and hour < hours:
                matrix[day, hour] = action_map.get(action, 0)
                price_matrix[day, hour] = price
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 动作热力图
        cmap_actions = plt.cm.RdYlGn_r
        im1 = axes[0].imshow(matrix, cmap=cmap_actions, aspect='auto', vmin=-1, vmax=1)
        axes[0].set_xlabel('Hour')
        axes[0].set_ylabel('Day')
        axes[0].set_title(f'{agent_name} - Actions')
        
        # 自定义颜色条
        cbar1 = plt.colorbar(im1, ax=axes[0], ticks=[-1, 0, 1])
        cbar1.ax.set_yticklabels(['Charge', 'Hold', 'Discharge'])
        
        # 价格热力图
        im2 = axes[1].imshow(price_matrix, cmap='YlOrRd', aspect='auto')
        axes[1].set_xlabel('Hour')
        axes[1].set_ylabel('Day')
        axes[1].set_title('Electricity Price ($/kWh)')
        plt.colorbar(im2, ax=axes[1], label='Price')
        
        plt.suptitle(f'Decision Pattern Analysis: {agent_name}')
        plt.tight_layout()
        self._save_figure(save_name)
        plt.show()
    
    def plot_strategy_evolution(
        self,
        reflexion_results: List,
        save_name: str = "strategy_evolution"
    ):
        """
        策略演化图
        展示 Reflexion Agent 如何随时间改进
        """
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # 1. 每日利润趋势
        ax1 = axes[0, 0]
        for i, run in enumerate(reflexion_results[:3]):  # 展示前3次运行
            daily = run.get('daily_profits', [])
            ax1.plot(range(1, len(daily)+1), daily, 'o-', alpha=0.7, label=f'Run {i+1}')
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Daily Profit ($)')
        ax1.set_title('Daily Profit Trend')
        ax1.legend()
        
        # 2. 动作比例演化
        ax2 = axes[0, 1]
        action_ratios = []
        for run in reflexion_results:
            history = run.get('history', [])
            for day in range(len(history) // 24):
                day_history = history[day*24:(day+1)*24]
                charge = sum(1 for h in day_history if h.get('action') == 'CHARGE')
                discharge = sum(1 for h in day_history if h.get('action') == 'DISCHARGE')
                hold = sum(1 for h in day_history if h.get('action') == 'HOLD')
                total = len(day_history)
                action_ratios.append({
                    'Day': day + 1,
                    'Charge': charge/total,
                    'Discharge': discharge/total,
                    'Hold': hold/total
                })
        
        ratio_df = pd.DataFrame(action_ratios).groupby('Day').mean()
        ratio_df.plot(kind='bar', stacked=True, ax=ax2, color=['green', 'red', 'gray'], alpha=0.7)
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Action Ratio')
        ax2.set_title('Action Distribution Evolution')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        ax2.legend(loc='upper right')
        
        # 3. 累积学习曲线（跨运行）
        ax3 = axes[1, 0]
        first_day_profits = [run.get('daily_profits', [0])[0] for run in reflexion_results]
        last_day_profits = [run.get('daily_profits', [0])[-1] for run in reflexion_results]
        
        x = range(1, len(reflexion_results) + 1)
        ax3.bar([i - 0.2 for i in x], first_day_profits, 0.4, label='Day 1', color='lightcoral')
        ax3.bar([i + 0.2 for i in x], last_day_profits, 0.4, label='Last Day', color='seagreen')
        ax3.set_xlabel('Run')
        ax3.set_ylabel('Profit ($)')
        ax3.set_title('First vs Last Day Profit')
        ax3.legend()
        
        # 4. 改进率
        ax4 = axes[1, 1]
        improvement_rates = []
        for run in reflexion_results:
            daily = run.get('daily_profits', [])
            if len(daily) >= 2:
                first_half = np.mean(daily[:len(daily)//2])
                second_half = np.mean(daily[len(daily)//2:])
                if first_half != 0:
                    improvement = (second_half - first_half) / abs(first_half) * 100
                else:
                    improvement = 0
                improvement_rates.append(improvement)
        
        ax4.bar(range(1, len(improvement_rates)+1), improvement_rates, color='steelblue')
        ax4.axhline(y=0, color='gray', linestyle='--')
        ax4.set_xlabel('Run')
        ax4.set_ylabel('Improvement Rate (%)')
        ax4.set_title('Second Half vs First Half Improvement')
        
        plt.suptitle('Reflexion Agent Strategy Evolution', fontsize=14)
        plt.tight_layout()
        self._save_figure(save_name)
        plt.show()
    
    def plot_statistical_comparison(
        self,
        results: Dict[str, List],
        baseline: str = 'RuleAgent',
        save_name: str = "statistical_comparison"
    ):
        """
        统计比较图
        展示置信区间和显著性
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        agents = [k for k in results.keys() if k != baseline]
        baseline_profits = [r.get('total_profit', 0) for r in results.get(baseline, [])]
        
        positions = []
        labels = []
        
        for i, agent in enumerate(agents):
            agent_profits = [r.get('total_profit', 0) for r in results.get(agent, [])]
            
            # 计算差异
            diff = [a - b for a, b in zip(agent_profits, baseline_profits)]
            mean_diff = np.mean(diff)
            
            # Bootstrap CI
            n_bootstrap = 10000
            boot_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(diff, size=len(diff), replace=True)
                boot_means.append(np.mean(sample))
            
            ci_lower = np.percentile(boot_means, 2.5)
            ci_upper = np.percentile(boot_means, 97.5)
            
            # t 检验
            t_stat, p_value = stats.ttest_rel(agent_profits, baseline_profits)
            
            # 绘制
            color = 'green' if ci_lower > 0 else ('red' if ci_upper < 0 else 'gray')
            
            ax.errorbar(
                mean_diff, i,
                xerr=[[mean_diff - ci_lower], [ci_upper - mean_diff]],
                fmt='o', color=color, markersize=10, capsize=5, capthick=2
            )
            
            # 添加显著性标记
            sig = ''
            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            
            ax.text(
                ci_upper + 0.5, i,
                f'p={p_value:.3f} {sig}',
                va='center', fontsize=9
            )
            
            labels.append(AGENT_LABELS.get(agent, agent))
            positions.append(i)
        
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel(f'Profit Difference vs {AGENT_LABELS.get(baseline, baseline)} ($)')
        ax.set_title('Statistical Comparison with 95% CI')
        
        # 添加注释
        ax.text(
            0.02, 0.98,
            '* p<0.05, ** p<0.01, *** p<0.001',
            transform=ax.transAxes,
            fontsize=8,
            va='top'
        )
        
        plt.tight_layout()
        self._save_figure(save_name)
        plt.show()
    
    def plot_price_action_overlay(
        self,
        history: List[Dict],
        price_data: pd.DataFrame,
        days_to_show: int = 3,
        save_name: str = "price_action_overlay"
    ):
        """
        价格与动作叠加图
        直观展示决策时机
        """
        hours_to_show = days_to_show * 24
        history = history[:hours_to_show]
        prices = price_data['price'].values[:hours_to_show]
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # 价格曲线
        hours = range(len(prices))
        ax.plot(hours, prices, 'b-', linewidth=1.5, label='Price', alpha=0.8)
        
        # 动作标记
        for i, h in enumerate(history):
            action = h.get('action', 'HOLD')
            if action == 'CHARGE':
                ax.axvspan(i, i+1, color='green', alpha=0.3)
            elif action == 'DISCHARGE':
                ax.axvspan(i, i+1, color='red', alpha=0.3)
        
        # SOC 曲线（次坐标轴）
        ax2 = ax.twinx()
        socs = [h.get('soc_after', 50) for h in history]
        ax2.plot(hours[:len(socs)], socs, 'g--', linewidth=1, label='SOC', alpha=0.7)
        ax2.set_ylabel('SOC (%)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim(0, 100)
        
        ax.set_xlabel('Hour')
        ax.set_ylabel('Price ($/kWh)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_title(f'Price and Action Timeline ({days_to_show} days)')
        
        # 自定义图例
        legend_elements = [
            Line2D([0], [0], color='blue', label='Price'),
            Line2D([0], [0], color='green', linestyle='--', label='SOC'),
            Patch(facecolor='green', alpha=0.3, label='Charge'),
            Patch(facecolor='red', alpha=0.3, label='Discharge'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 日分隔
        for day in range(1, days_to_show + 1):
            ax.axvline(x=day * 24, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        self._save_figure(save_name)
        plt.show()
    
    def plot_ablation_results(
        self,
        ablation_results: Dict[str, Dict],
        metric: str = 'total_profit',
        save_name: str = "ablation"
    ):
        """
        消融实验结果图
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        exp_names = list(ablation_results.keys())
        
        # 提取 Reflexion Agent 的结果
        reflexion_means = []
        reflexion_stds = []
        
        for exp_name in exp_names:
            exp_results = ablation_results[exp_name]
            if 'reflexion' in exp_results:
                profits = [r.total_profit for r in exp_results['reflexion']]
                reflexion_means.append(np.mean(profits))
                reflexion_stds.append(np.std(profits))
            else:
                reflexion_means.append(0)
                reflexion_stds.append(0)
        
        x = range(len(exp_names))
        bars = ax.bar(x, reflexion_means, yerr=reflexion_stds, capsize=5,
                      color='steelblue', alpha=0.8)
        
        # 高亮最佳
        best_idx = np.argmax(reflexion_means)
        bars[best_idx].set_color('green')
        
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.set_ylabel(f'{metric.replace("_", " ").title()} ($)')
        ax.set_title('Ablation Study Results')
        ax.axhline(y=reflexion_means[0], color='gray', linestyle='--', alpha=0.5,
                   label='Baseline')
        
        plt.tight_layout()
        self._save_figure(save_name)
        plt.show()
    
    def create_paper_figure(
        self,
        results: Dict[str, List],
        price_data: pd.DataFrame,
        save_name: str = "paper_main_figure"
    ):
        """
        创建论文主图（多面板组合）
        """
        fig = plt.figure(figsize=(14, 10))
        
        # 布局: 2x2
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # (a) 累积利润
        ax1 = fig.add_subplot(gs[0, 0])
        for agent_name, runs in results.items():
            profits = runs[0].get('hourly_profits', []) if runs else []
            cumulative = np.cumsum(profits)
            color = COLORS.get(agent_name.lower().replace('agent', ''), '#333')
            label = AGENT_LABELS.get(agent_name, agent_name)
            ax1.plot(cumulative, label=label, color=color, linewidth=2)
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Cumulative Profit ($)')
        ax1.set_title('(a) Cumulative Profit')
        ax1.legend(loc='upper left')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # (b) 每日利润条形图
        ax2 = fig.add_subplot(gs[0, 1])
        n_agents = len(results)
        width = 0.8 / n_agents
        for idx, (agent_name, runs) in enumerate(results.items()):
            daily = runs[0].get('daily_profits', []) if runs else []
            x = np.arange(len(daily))
            offset = (idx - n_agents/2 + 0.5) * width
            color = COLORS.get(agent_name.lower().replace('agent', ''), '#333')
            label = AGENT_LABELS.get(agent_name, agent_name)
            ax2.bar(x + offset, daily, width, label=label, color=color, alpha=0.8)
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Daily Profit ($)')
        ax2.set_title('(b) Daily Profit')
        ax2.legend()
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # (c) 动作分布
        ax3 = fig.add_subplot(gs[1, 0])
        action_data = []
        for agent_name, runs in results.items():
            history = runs[0].get('history', []) if runs else []
            charge = sum(1 for h in history if h.get('action') == 'CHARGE')
            discharge = sum(1 for h in history if h.get('action') == 'DISCHARGE')
            hold = sum(1 for h in history if h.get('action') == 'HOLD')
            total = len(history) or 1
            action_data.append({
                'Agent': AGENT_LABELS.get(agent_name, agent_name),
                'Charge': charge/total*100,
                'Discharge': discharge/total*100,
                'Hold': hold/total*100
            })
        
        action_df = pd.DataFrame(action_data)
        action_df.set_index('Agent')[['Charge', 'Discharge', 'Hold']].plot(
            kind='bar', stacked=True, ax=ax3,
            color=['green', 'red', 'gray'], alpha=0.8
        )
        ax3.set_ylabel('Action Ratio (%)')
        ax3.set_title('(c) Action Distribution')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.legend(loc='upper right')
        
        # (d) 价格与 SOC
        ax4 = fig.add_subplot(gs[1, 1])
        if 'ReflexionAgent' in results and results['ReflexionAgent']:
            history = results['ReflexionAgent'][0].get('history', [])[:72]  # 3天
            prices = price_data['price'].values[:72]
            
            ax4.plot(range(len(prices)), prices, 'b-', label='Price', linewidth=1.5)
            
            ax4_twin = ax4.twinx()
            socs = [h.get('soc_after', 50) for h in history]
            ax4_twin.plot(range(len(socs)), socs, 'g--', label='SOC', alpha=0.7)
            ax4_twin.set_ylabel('SOC (%)', color='green')
            ax4_twin.set_ylim(0, 100)
            
            ax4.set_xlabel('Hour')
            ax4.set_ylabel('Price ($/kWh)', color='blue')
            ax4.set_title('(d) Price and SOC Profile (3 days)')
        
        plt.suptitle('Battery Arbitrage Agent Performance Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        self._save_figure(save_name)
        plt.show()
    
    def _save_figure(self, name: str):
        """保存图表为多种格式"""
        for fmt in ['png', 'pdf', 'svg']:
            path = os.path.join(self.output_dir, f"{name}.{fmt}")
            plt.savefig(path, format=fmt, bbox_inches='tight', dpi=300)
        print(f"📊 Figure saved: {self.output_dir}/{name}.[png/pdf/svg]")
