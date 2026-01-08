"""
实验运行器
支持多次运行、消融实验、结果聚合
"""
import os
import json
import yaml
import copy
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict

from src.env import BatteryEnv
from src.agents import RuleAgent, SimpleLLMAgent, ReflexionAgent, BaseAgent, DEFAULT_MODEL
from src.metrics import MetricsCalculator, calculate_theoretical_optimal
from src.utils import load_market_data


@dataclass
class ExperimentResult:
    """单次实验结果"""
    agent_name: str
    run_id: int
    seed: int
    total_profit: float
    hourly_profits: List[float]
    daily_profits: List[float]
    history: List[Dict]
    llm_calls: int
    metrics: Dict = field(default_factory=dict)
    config: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ExperimentRunner:
    """
    实验运行器
    
    功能:
    - 多次运行取平均
    - 消融实验
    - 随机种子控制
    - 结果聚合和统计
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None,
        output_dir: str = "outputs"
    ):
        """
        Args:
            config_path: YAML 配置文件路径
            config: 直接传入的配置字典
            output_dir: 输出目录
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            self.config = self._default_config()
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.metrics_calc = MetricsCalculator()
        self.results: Dict[str, List[ExperimentResult]] = {}
        
        # 实验时间戳
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'experiment': {
                'name': 'default',
                'seed': 42,
                'num_runs': 5,
                'num_days': 7
            },
            'agents': {
                'rule': {'enabled': True},
                'reflexion': {'enabled': True, 'model': DEFAULT_MODEL}
            }
        }
    
    def _create_agent(self, agent_type: str, config: Dict) -> BaseAgent:
        """根据配置创建 Agent"""
        if agent_type == 'rule':
            return RuleAgent()
        elif agent_type == 'simple_llm':
            return SimpleLLMAgent(
                model_name=config.get('model', DEFAULT_MODEL),
                temperature=config.get('temperature', 0.1)
            )
        elif agent_type == 'reflexion':
            return ReflexionAgent(
                model_name=config.get('model', DEFAULT_MODEL),
                temperature=config.get('decision_temperature', 0.1),
                reflection_temperature=config.get('reflection_temperature', 0.3)
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def run_single(
        self,
        agent: BaseAgent,
        df: pd.DataFrame,
        num_days: int,
        seed: int,
        run_id: int = 0
    ) -> ExperimentResult:
        """运行单次实验"""
        np.random.seed(seed)
        
        env = BatteryEnv(df)
        agent.reset()
        obs, _ = env.reset(seed=seed, options={"initial_soc": 0.5})
        
        hourly_profits = []
        daily_profits = []
        all_history = []
        
        for day in range(num_days):
            daily_buffer = []
            daily_profit = 0
            
            for hour in range(24):
                if obs is None:
                    break
                
                action = agent.decide(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                hourly_profits.append(reward)
                daily_profit += reward
                daily_buffer.append(info)
                all_history.append(info)
                
                if hasattr(agent, 'record_transaction'):
                    agent.record_transaction(info)
                
                obs = next_obs
                if done:
                    break
            
            daily_profits.append(daily_profit)
            agent.end_of_day(daily_buffer)
        
        # 计算详细指标
        metrics = self.metrics_calc.generate_full_report(
            hourly_profits, all_history, agent.name
        )
        
        return ExperimentResult(
            agent_name=agent.name,
            run_id=run_id,
            seed=seed,
            total_profit=sum(hourly_profits),
            hourly_profits=hourly_profits,
            daily_profits=daily_profits,
            history=all_history,
            llm_calls=getattr(agent, 'total_llm_calls', 0),
            metrics=metrics
        )
    
    def run_multiple(
        self,
        agent_type: str,
        agent_config: Dict,
        df: pd.DataFrame,
        num_days: int,
        num_runs: int,
        base_seed: int
    ) -> List[ExperimentResult]:
        """多次运行同一 Agent"""
        results = []
        
        for run_id in range(num_runs):
            seed = base_seed + run_id * 1000
            agent = self._create_agent(agent_type, agent_config)
            
            result = self.run_single(
                agent=agent,
                df=df,
                num_days=num_days,
                seed=seed,
                run_id=run_id
            )
            result.config = agent_config
            results.append(result)
            
            print(f"  Run {run_id + 1}/{num_runs}: Profit = ${result.total_profit:.4f}")
        
        return results
    
    def run_experiment(self, verbose: bool = True) -> Dict[str, List[ExperimentResult]]:
        """运行完整实验"""
        exp_config = self.config.get('experiment', {})
        num_runs = exp_config.get('num_runs', 5)
        num_days = exp_config.get('num_days', 7)
        base_seed = exp_config.get('seed', 42)
        
        # 加载数据
        df = load_market_data()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔬 Experiment: {exp_config.get('name', 'default')}")
            print(f"📊 Data: {len(df)} hours")
            print(f"🔄 Runs: {num_runs}")
            print(f"📅 Days: {num_days}")
            print(f"{'='*60}\n")
        
        results = {}
        
        # 运行各 Agent
        agents_config = self.config.get('agents', {})
        
        for agent_type, agent_config in agents_config.items():
            if not agent_config.get('enabled', True):
                continue
            
            if verbose:
                print(f"\n🤖 Running: {agent_type}")
                print("-" * 40)
            
            agent_results = self.run_multiple(
                agent_type=agent_type,
                agent_config=agent_config,
                df=df,
                num_days=num_days,
                num_runs=num_runs,
                base_seed=base_seed
            )
            
            results[agent_type] = agent_results
            
            # 打印汇总
            profits = [r.total_profit for r in agent_results]
            if verbose:
                print(f"\n  📈 Mean Profit: ${np.mean(profits):.4f} ± ${np.std(profits):.4f}")
        
        self.results = results
        return results
    
    def run_ablation(self, verbose: bool = True) -> Dict[str, Dict[str, List[ExperimentResult]]]:
        """运行消融实验"""
        ablation_config = self.config.get('ablation', {})
        
        if not ablation_config.get('enabled', False):
            print("Ablation experiments not enabled in config")
            return {}
        
        experiments = ablation_config.get('experiments', [])
        ablation_results = {}
        
        df = load_market_data()
        exp_config = self.config.get('experiment', {})
        num_runs = exp_config.get('num_runs', 3)  # 消融实验可以少跑几次
        num_days = exp_config.get('num_days', 7)
        base_seed = exp_config.get('seed', 42)
        
        for exp in experiments:
            exp_name = exp.get('name', 'unnamed')
            modifications = exp.get('modify', {})
            
            if verbose:
                print(f"\n🔬 Ablation: {exp_name}")
                print(f"   Modifications: {modifications}")
            
            # 创建修改后的配置
            modified_config = copy.deepcopy(self.config)
            for key_path, value in modifications.items():
                self._set_nested(modified_config, key_path, value)
            
            # 运行实验
            agents_config = modified_config.get('agents', {})
            exp_results = {}
            
            for agent_type, agent_config in agents_config.items():
                if not agent_config.get('enabled', True):
                    continue
                
                agent_results = self.run_multiple(
                    agent_type=agent_type,
                    agent_config=agent_config,
                    df=df,
                    num_days=num_days,
                    num_runs=num_runs,
                    base_seed=base_seed
                )
                exp_results[agent_type] = agent_results
            
            ablation_results[exp_name] = exp_results
        
        return ablation_results
    
    def _set_nested(self, d: Dict, key_path: str, value: Any):
        """设置嵌套字典的值"""
        keys = key_path.split('.')
        current = d
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value
    
    def aggregate_results(self) -> pd.DataFrame:
        """聚合多次运行结果"""
        rows = []
        
        for agent_name, agent_results in self.results.items():
            profits = [r.total_profit for r in agent_results]
            sharpes = [r.metrics['financial']['sharpe_ratio'] for r in agent_results]
            win_rates = [r.metrics['financial']['win_rate'] for r in agent_results]
            max_dds = [r.metrics['financial']['max_drawdown'] for r in agent_results]
            
            row = {
                'Agent': agent_name,
                'Profit (Mean)': np.mean(profits),
                'Profit (Std)': np.std(profits),
                'Profit (Min)': np.min(profits),
                'Profit (Max)': np.max(profits),
                'Sharpe (Mean)': np.mean(sharpes),
                'Sharpe (Std)': np.std(sharpes),
                'Win Rate (Mean)': np.mean(win_rates),
                'Max DD (Mean)': np.mean(max_dds),
                'Runs': len(agent_results),
                'LLM Calls (Total)': sum(r.llm_calls for r in agent_results)
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def statistical_comparison(
        self,
        baseline: str = 'rule'
    ) -> pd.DataFrame:
        """统计显著性比较"""
        if baseline not in self.results:
            print(f"Baseline {baseline} not found")
            return pd.DataFrame()
        
        baseline_profits = [r.total_profit for r in self.results[baseline]]
        
        rows = []
        for agent_name, agent_results in self.results.items():
            if agent_name == baseline:
                continue
            
            agent_profits = [r.total_profit for r in agent_results]
            
            # t 检验
            t_result = self.metrics_calc.paired_t_test(
                agent_profits, baseline_profits, 'greater'
            )
            
            # Wilcoxon 检验
            w_result = self.metrics_calc.wilcoxon_test(
                agent_profits, baseline_profits, 'greater'
            )
            
            # Bootstrap CI
            diff = [a - b for a, b in zip(agent_profits, baseline_profits)]
            mean_diff, ci_lower, ci_upper = self.metrics_calc.bootstrap_confidence_interval(diff)
            
            rows.append({
                'Agent': agent_name,
                'vs': baseline,
                'Mean Diff': mean_diff,
                'CI Lower': ci_lower,
                'CI Upper': ci_upper,
                't-stat': t_result['t_stat'],
                'p-value (t)': t_result['p_value'],
                "Cohen's d": t_result['cohens_d'],
                'p-value (Wilcoxon)': w_result['p_value'],
                'Significant (α=0.05)': t_result['p_value'] < 0.05
            })
        
        return pd.DataFrame(rows)
    
    def save_results(self, prefix: str = ""):
        """保存所有结果"""
        timestamp = self.experiment_id
        
        # 保存聚合结果
        agg_df = self.aggregate_results()
        agg_df.to_csv(
            os.path.join(self.output_dir, f"{prefix}aggregated_{timestamp}.csv"),
            index=False
        )
        
        # 保存统计比较
        if 'rule' in self.results:
            stat_df = self.statistical_comparison('rule')
            stat_df.to_csv(
                os.path.join(self.output_dir, f"{prefix}statistics_{timestamp}.csv"),
                index=False
            )
        
        # 保存详细结果
        detailed = {}
        for agent_name, agent_results in self.results.items():
            detailed[agent_name] = [
                {
                    'run_id': r.run_id,
                    'seed': r.seed,
                    'total_profit': r.total_profit,
                    'llm_calls': r.llm_calls,
                    'metrics': r.metrics
                }
                for r in agent_results
            ]
        
        with open(os.path.join(self.output_dir, f"{prefix}detailed_{timestamp}.json"), 'w') as f:
            json.dump(detailed, f, indent=2, default=str)
        
        # 保存配置
        with open(os.path.join(self.output_dir, f"{prefix}config_{timestamp}.yaml"), 'w') as f:
            yaml.dump(self.config, f)
        
        print(f"\n💾 Results saved to {self.output_dir}/")
    
    def generate_latex_table(self) -> str:
        """生成 LaTeX 格式的结果表格（适合论文）"""
        df = self.aggregate_results()
        
        latex = r"""
\begin{table}[h]
\centering
\caption{Performance Comparison of Battery Arbitrage Agents}
\label{tab:results}
\begin{tabular}{lccccc}
\toprule
Agent & Profit (\$) & Sharpe Ratio & Win Rate & Max DD (\$) & LLM Calls \\
\midrule
"""
        
        for _, row in df.iterrows():
            profit_str = f"${row['Profit (Mean)']:.2f} \\pm {row['Profit (Std)']:.2f}$"
            sharpe_str = f"{row['Sharpe (Mean)']:.3f}"
            win_str = f"{row['Win Rate (Mean)']*100:.1f}\\%"
            dd_str = f"${row['Max DD (Mean)']:.2f}$"
            llm_str = str(int(row['LLM Calls (Total)']))
            
            latex += f"{row['Agent']} & {profit_str} & {sharpe_str} & {win_str} & {dd_str} & {llm_str} \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return latex


def run_full_experiment(config_path: str = "configs/default.yaml"):
    """便捷函数：运行完整实验"""
    runner = ExperimentRunner(config_path=config_path)
    
    # 主实验
    results = runner.run_experiment(verbose=True)
    
    # 打印汇总
    print("\n" + "="*60)
    print("📊 AGGREGATED RESULTS")
    print("="*60)
    print(runner.aggregate_results().to_string())
    
    # 统计比较
    if 'rule' in results:
        print("\n" + "="*60)
        print("📈 STATISTICAL COMPARISON (vs RuleAgent)")
        print("="*60)
        print(runner.statistical_comparison().to_string())
    
    # 保存
    runner.save_results()
    
    # LaTeX 表格
    print("\n" + "="*60)
    print("📝 LATEX TABLE")
    print("="*60)
    print(runner.generate_latex_table())
    
    return runner


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--ablation', action='store_true')
    args = parser.parse_args()
    
    runner = run_full_experiment(args.config)
    
    if args.ablation:
        print("\n" + "="*60)
        print("🔬 RUNNING ABLATION EXPERIMENTS")
        print("="*60)
        runner.run_ablation(verbose=True)
