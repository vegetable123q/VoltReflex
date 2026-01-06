"""
评估指标模块
包含金融指标、操作指标和统计检验
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import pandas as pd


class MetricsCalculator:
    """
    计算各类评估指标
    用于学术论文的定量分析
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: 年化无风险利率 (默认 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.hourly_rf = risk_free_rate / (365 * 24)  # 小时化
    
    # ============================================================
    # 金融指标
    # ============================================================
    
    def calculate_sharpe_ratio(
        self, 
        returns: List[float], 
        annualize: bool = True
    ) -> float:
        """
        计算夏普比率 (Sharpe Ratio)
        
        Sharpe = (E[R] - Rf) / σ[R]
        
        Args:
            returns: 收益序列
            annualize: 是否年化
            
        Returns:
            夏普比率
        """
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        excess_returns = returns - self.hourly_rf
        
        mean_excess = np.mean(excess_returns)
        std_returns = np.std(returns, ddof=1)
        
        if std_returns == 0:
            return 0.0
        
        sharpe = mean_excess / std_returns
        
        if annualize:
            # 年化：假设一年 8760 小时
            sharpe *= np.sqrt(8760)
        
        return float(sharpe)
    
    def calculate_sortino_ratio(
        self, 
        returns: List[float],
        annualize: bool = True
    ) -> float:
        """
        计算索提诺比率 (Sortino Ratio)
        
        只考虑下行波动率，更适合评估风险
        Sortino = (E[R] - Rf) / σ[R_negative]
        """
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        excess_returns = returns - self.hourly_rf
        
        mean_excess = np.mean(excess_returns)
        
        # 只计算负收益的标准差
        negative_returns = returns[returns < 0]
        if len(negative_returns) < 2:
            return float('inf') if mean_excess > 0 else 0.0
        
        downside_std = np.std(negative_returns, ddof=1)
        
        if downside_std == 0:
            return float('inf') if mean_excess > 0 else 0.0
        
        sortino = mean_excess / downside_std
        
        if annualize:
            sortino *= np.sqrt(8760)
        
        return float(sortino)
    
    def calculate_max_drawdown(
        self, 
        cumulative_profits: List[float]
    ) -> Tuple[float, int, int]:
        """
        计算最大回撤 (Maximum Drawdown)
        
        Args:
            cumulative_profits: 累积收益序列
            
        Returns:
            (最大回撤值, 回撤开始索引, 回撤结束索引)
        """
        if len(cumulative_profits) < 2:
            return 0.0, 0, 0
        
        cumulative = np.array(cumulative_profits)
        
        # 计算运行最大值
        running_max = np.maximum.accumulate(cumulative)
        
        # 计算回撤
        drawdown = running_max - cumulative
        
        max_dd = np.max(drawdown)
        max_dd_end = np.argmax(drawdown)
        max_dd_start = np.argmax(cumulative[:max_dd_end + 1]) if max_dd_end > 0 else 0
        
        return float(max_dd), int(max_dd_start), int(max_dd_end)
    
    def calculate_profit_factor(
        self, 
        returns: List[float]
    ) -> float:
        """
        计算盈亏比 (Profit Factor)
        
        Profit Factor = 总盈利 / 总亏损
        > 1 表示盈利系统
        """
        returns = np.array(returns)
        
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return float(gross_profit / gross_loss)
    
    def calculate_win_rate(self, returns: List[float]) -> float:
        """计算胜率"""
        returns = np.array(returns)
        if len(returns) == 0:
            return 0.0
        return float(np.sum(returns > 0) / len(returns))
    
    def calculate_calmar_ratio(
        self,
        returns: List[float],
        cumulative_profits: List[float]
    ) -> float:
        """
        计算卡玛比率 (Calmar Ratio)
        
        Calmar = 年化收益率 / 最大回撤
        """
        total_return = sum(returns)
        max_dd, _, _ = self.calculate_max_drawdown(cumulative_profits)
        
        if max_dd == 0:
            return float('inf') if total_return > 0 else 0.0
        
        # 假设 7 天数据，年化
        annualized_return = total_return * (365 / 7)
        
        return float(annualized_return / max_dd)
    
    # ============================================================
    # 操作指标
    # ============================================================
    
    def calculate_cycle_count(self, history: List[Dict]) -> float:
        """
        计算等效满充放电次数
        
        一个完整循环 = 充满后完全放空
        """
        total_charged = sum(h.get('energy_charged', 0) for h in history)
        total_discharged = sum(h.get('energy_discharged', 0) for h in history)
        
        # 等效循环 = 平均(充电量, 放电量) / 电池容量
        avg_throughput = (total_charged + total_discharged) / 2
        capacity = 13.5  # Tesla Powerwall
        
        return float(avg_throughput / capacity)
    
    def calculate_soc_statistics(
        self, 
        history: List[Dict]
    ) -> Dict[str, float]:
        """计算 SOC 统计指标"""
        socs = [h.get('soc_after', 50) for h in history]
        
        if not socs:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        return {
            'mean': float(np.mean(socs)),
            'std': float(np.std(socs)),
            'min': float(np.min(socs)),
            'max': float(np.max(socs))
        }
    
    def calculate_price_capture_ratio(
        self,
        history: List[Dict],
        optimal_profit: Optional[float] = None
    ) -> float:
        """
        计算价格捕获率
        
        实际利润 / 理论最优利润
        """
        actual_profit = sum(h.get('reward', 0) for h in history)
        
        if optimal_profit is None:
            # 简单估算理论最优：假设总是在最低价充电、最高价放电
            prices = [h['price'] for h in history]
            if not prices:
                return 0.0
            
            # 粗略估计：(最高价 - 最低价) * 每天可交易容量 * 效率
            price_spread = max(prices) - min(prices)
            days = len(history) / 24
            optimal_profit = price_spread * 13.5 * 0.9 * days
        
        if optimal_profit <= 0:
            return 0.0
        
        return float(actual_profit / optimal_profit)
    
    def calculate_action_entropy(self, history: List[Dict]) -> float:
        """
        计算动作熵
        
        衡量策略的多样性/随机性
        """
        actions = [h.get('action', 'HOLD') for h in history]
        
        if not actions:
            return 0.0
        
        # 计算动作分布
        action_counts = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        
        total = len(actions)
        probs = [count / total for count in action_counts.values()]
        
        # 计算熵
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        # 归一化到 [0, 1]（最大熵为 log2(3) ≈ 1.58）
        max_entropy = np.log2(3)
        
        return float(entropy / max_entropy)
    
    # ============================================================
    # 统计检验
    # ============================================================
    
    def paired_t_test(
        self,
        returns_a: List[float],
        returns_b: List[float],
        alternative: str = 'greater'
    ) -> Dict[str, float]:
        """
        配对 t 检验
        
        检验 Agent A 是否显著优于 Agent B
        
        Args:
            returns_a: Agent A 的日收益
            returns_b: Agent B 的日收益
            alternative: 'greater', 'less', or 'two-sided'
            
        Returns:
            包含 t统计量, p值, 效应量 的字典
        """
        a = np.array(returns_a)
        b = np.array(returns_b)
        
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a, b = a[:min_len], b[:min_len]
        
        if len(a) < 2:
            return {'t_stat': 0, 'p_value': 1.0, 'cohens_d': 0}
        
        # 配对 t 检验
        t_stat, p_value = stats.ttest_rel(a, b, alternative=alternative)
        
        # Cohen's d 效应量
        diff = a - b
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
        
        return {
            't_stat': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d)
        }
    
    def wilcoxon_test(
        self,
        returns_a: List[float],
        returns_b: List[float],
        alternative: str = 'greater'
    ) -> Dict[str, float]:
        """
        Wilcoxon 符号秩检验（非参数检验）
        
        当数据不满足正态分布假设时使用
        """
        a = np.array(returns_a)
        b = np.array(returns_b)
        
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a, b = a[:min_len], b[:min_len]
        
        diff = a - b
        
        # 移除零差异
        diff = diff[diff != 0]
        
        if len(diff) < 2:
            return {'statistic': 0, 'p_value': 1.0}
        
        try:
            stat, p_value = stats.wilcoxon(diff, alternative=alternative)
            return {'statistic': float(stat), 'p_value': float(p_value)}
        except Exception:
            return {'statistic': 0, 'p_value': 1.0}
    
    def bootstrap_confidence_interval(
        self,
        returns: List[float],
        statistic: str = 'mean',
        confidence: float = 0.95,
        n_bootstrap: int = 10000
    ) -> Tuple[float, float, float]:
        """
        Bootstrap 置信区间
        
        Args:
            returns: 收益序列
            statistic: 'mean' 或 'median'
            confidence: 置信水平
            n_bootstrap: Bootstrap 采样次数
            
        Returns:
            (统计量, 下界, 上界)
        """
        returns = np.array(returns)
        n = len(returns)
        
        if n < 2:
            val = returns[0] if n == 1 else 0
            return val, val, val
        
        # Bootstrap 采样
        boot_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=n, replace=True)
            if statistic == 'mean':
                boot_stats.append(np.mean(sample))
            else:
                boot_stats.append(np.median(sample))
        
        boot_stats = np.array(boot_stats)
        
        # 计算置信区间
        alpha = 1 - confidence
        lower = np.percentile(boot_stats, 100 * alpha / 2)
        upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
        
        point_estimate = np.mean(returns) if statistic == 'mean' else np.median(returns)
        
        return float(point_estimate), float(lower), float(upper)
    
    # ============================================================
    # 综合报告
    # ============================================================
    
    def generate_full_report(
        self,
        hourly_profits: List[float],
        history: List[Dict],
        agent_name: str = "Agent"
    ) -> Dict:
        """
        生成完整的评估报告
        """
        # 累积利润
        cumulative = []
        total = 0
        for p in hourly_profits:
            total += p
            cumulative.append(total)
        
        # 日利润
        daily_profits = []
        for day in range(len(hourly_profits) // 24):
            day_profit = sum(hourly_profits[day*24:(day+1)*24])
            daily_profits.append(day_profit)
        
        # 计算所有指标
        max_dd, dd_start, dd_end = self.calculate_max_drawdown(cumulative)
        soc_stats = self.calculate_soc_statistics(history)
        
        report = {
            'agent': agent_name,
            'financial': {
                'total_profit': sum(hourly_profits),
                'mean_hourly_profit': np.mean(hourly_profits),
                'mean_daily_profit': np.mean(daily_profits) if daily_profits else 0,
                'sharpe_ratio': self.calculate_sharpe_ratio(hourly_profits),
                'sortino_ratio': self.calculate_sortino_ratio(hourly_profits),
                'max_drawdown': max_dd,
                'max_drawdown_period': (dd_start, dd_end),
                'profit_factor': self.calculate_profit_factor(hourly_profits),
                'calmar_ratio': self.calculate_calmar_ratio(hourly_profits, cumulative),
                'win_rate': self.calculate_win_rate(hourly_profits),
            },
            'operational': {
                'total_trades': sum(1 for h in history if h.get('action') != 'HOLD'),
                'charge_count': sum(1 for h in history if h.get('action') == 'CHARGE'),
                'discharge_count': sum(1 for h in history if h.get('action') == 'DISCHARGE'),
                'hold_count': sum(1 for h in history if h.get('action') == 'HOLD'),
                'cycle_count': self.calculate_cycle_count(history),
                'soc_mean': soc_stats['mean'],
                'soc_std': soc_stats['std'],
                'action_entropy': self.calculate_action_entropy(history),
            },
            'efficiency': {
                'price_capture_ratio': self.calculate_price_capture_ratio(history),
                'total_energy_charged': sum(h.get('energy_charged', 0) for h in history),
                'total_energy_discharged': sum(h.get('energy_discharged', 0) for h in history),
            },
            'confidence_interval': self.bootstrap_confidence_interval(hourly_profits)
        }
        
        return report
    
    def compare_agents(
        self,
        results: Dict[str, Dict],
        baseline: str = 'RuleAgent'
    ) -> pd.DataFrame:
        """
        生成 Agent 对比表格（适合论文使用）
        """
        rows = []
        
        for agent_name, result in results.items():
            hourly = result.get('hourly_profits', [])
            history = result.get('history', [])
            
            report = self.generate_full_report(hourly, history, agent_name)
            
            row = {
                'Agent': agent_name,
                'Total Profit ($)': f"{report['financial']['total_profit']:.2f}",
                'Sharpe Ratio': f"{report['financial']['sharpe_ratio']:.3f}",
                'Sortino Ratio': f"{report['financial']['sortino_ratio']:.3f}",
                'Max Drawdown ($)': f"{report['financial']['max_drawdown']:.2f}",
                'Win Rate (%)': f"{report['financial']['win_rate']*100:.1f}",
                'Profit Factor': f"{report['financial']['profit_factor']:.2f}",
                'Cycles': f"{report['operational']['cycle_count']:.1f}",
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 添加统计显著性检验
        if baseline in results and len(results) > 1:
            baseline_daily = self._get_daily_profits(results[baseline])
            
            for agent_name in results:
                if agent_name != baseline:
                    agent_daily = self._get_daily_profits(results[agent_name])
                    test_result = self.paired_t_test(agent_daily, baseline_daily)
                    p_val = test_result['p_value']
                    
                    # 添加显著性标记
                    sig = ''
                    if p_val < 0.001:
                        sig = '***'
                    elif p_val < 0.01:
                        sig = '**'
                    elif p_val < 0.05:
                        sig = '*'
                    
                    idx = df[df['Agent'] == agent_name].index[0]
                    df.loc[idx, 'Total Profit ($)'] += sig
        
        return df
    
    def _get_daily_profits(self, result: Dict) -> List[float]:
        """从结果中提取日利润"""
        hourly = result.get('hourly_profits', [])
        daily = []
        for day in range(len(hourly) // 24):
            daily.append(sum(hourly[day*24:(day+1)*24]))
        return daily


def calculate_theoretical_optimal(df: pd.DataFrame, battery_capacity: float = 13.5) -> float:
    """
    计算理论最优利润（完美预知）
    
    使用动态规划找到最优充放电策略
    """
    prices = df['price'].values
    n = len(prices)
    max_power = 5.0
    efficiency = 0.9
    
    # 状态: SOC 离散化为 100 个等级
    n_soc = 101
    soc_levels = np.linspace(0, 1, n_soc)
    
    # DP: value[t][soc] = 从时刻 t 开始，初始 SOC 为 soc 时的最大收益
    value = np.full((n + 1, n_soc), -np.inf)
    value[n, :] = 0  # 终止状态
    
    for t in range(n - 1, -1, -1):
        price = prices[t]
        
        for i, soc in enumerate(soc_levels):
            current_energy = soc * battery_capacity
            
            # 动作1: HOLD
            hold_value = value[t + 1, i]
            
            # 动作2: CHARGE
            charge_value = -np.inf
            if soc < 0.95:
                charge_energy = min(max_power, (0.95 - soc) * battery_capacity)
                new_soc = soc + charge_energy / battery_capacity
                new_soc_idx = int(new_soc * 100)
                cost = price * charge_energy / np.sqrt(efficiency)
                if new_soc_idx < n_soc:
                    charge_value = -cost + value[t + 1, new_soc_idx]
            
            # 动作3: DISCHARGE
            discharge_value = -np.inf
            if soc > 0.15:
                discharge_energy = min(max_power, (soc - 0.1) * battery_capacity)
                new_soc = soc - discharge_energy / battery_capacity
                new_soc_idx = int(new_soc * 100)
                revenue = price * discharge_energy * np.sqrt(efficiency)
                if new_soc_idx >= 0:
                    discharge_value = revenue + value[t + 1, new_soc_idx]
            
            value[t, i] = max(hold_value, charge_value, discharge_value)
    
    # 从 50% SOC 开始
    optimal_profit = value[0, 50]
    
    return float(optimal_profit)
