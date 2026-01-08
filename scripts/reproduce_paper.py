"""
论文结果复现脚本
一键生成所有实验结果和图表
"""
import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import BatteryEnv
from src.agents import RuleAgent, SimpleLLMAgent, ReflexionAgent
from src.metrics import MetricsCalculator
from src.data_loader import RealDataLoader
from src.experiment import ExperimentRunner
from src.visualization import AcademicVisualizer
from src.rl_baselines import SimpleQAgent, DQNAgent, MPCBaseline, train_rl_agent


def load_config(config_path: str = None) -> dict:
    """加载配置"""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs', 'default.yaml'
        )
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_output_dirs(base_dir: str) -> dict:
    """创建输出目录"""
    dirs = {
        'base': base_dir,
        'figures': os.path.join(base_dir, 'figures'),
        'tables': os.path.join(base_dir, 'tables'),
        'data': os.path.join(base_dir, 'data'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    return dirs


def run_main_experiments(config: dict, output_dirs: dict) -> dict:
    """
    实验1: 主实验 - 三种策略对比
    """
    print("\n" + "="*60)
    print("Experiment 1: Main Comparison (Rule vs Zero-shot vs Reflexion)")
    print("="*60)
    
    # 加载数据
    loader = RealDataLoader()
    data_source = config['data']['source']
    
    if data_source == 'synthetic':
        from src.utils import load_market_data
        data = load_market_data(config['data']['path'])
    else:
        data = loader.load_caiso_data(
            config['data']['start_date'],
            config['data']['end_date']
        )
    
    # 初始化实验运行器
    runner = ExperimentRunner(
        config=config,
        output_dir=output_dirs['logs']
    )
    
    results = {}
    n_runs = config['experiment'].get('n_runs', 5)
    
    # 1. Rule-based Agent
    print("\nRunning Rule-based Agent...")
    rule_results = runner.run_multiple_experiments(
        agent_class=RuleAgent,
        agent_config={},
        data=data,
        n_runs=n_runs,
        agent_name='rule'
    )
    results['rule'] = rule_results
    
    # 2. Zero-shot LLM Agent (如果配置了 API key)
    if os.getenv('OPENAI_API_KEY'):
        print("\nRunning Zero-shot LLM Agent...")
        llm_results = runner.run_multiple_experiments(
            agent_class=SimpleLLMAgent,
            agent_config={'model': config['agents']['llm']['model']},
            data=data,
            n_runs=n_runs,
            agent_name='zero_shot'
        )
        results['zero_shot'] = llm_results
        
        # 3. Reflexion Agent
        print("\nRunning Reflexion Agent...")
        reflexion_results = runner.run_multiple_experiments(
            agent_class=ReflexionAgent,
            agent_config={'model': config['agents']['llm']['model']},
            data=data,
            n_runs=n_runs,
            agent_name='reflexion'
        )
        results['reflexion'] = reflexion_results
    else:
        print("\n[Warning] OPENAI_API_KEY not set, skipping LLM agents")
    
    # 保存结果
    results_path = os.path.join(output_dirs['data'], 'main_experiment_results.json')
    
    # 转换为可序列化格式
    serializable_results = {}
    for agent_name, agent_results in results.items():
        serializable_results[agent_name] = {
            'summary': agent_results.get('summary', {}),
            'total_profits': agent_results.get('total_profits', []),
            'daily_returns': [
                r if isinstance(r, list) else r.tolist() 
                for r in agent_results.get('daily_returns', [])
            ]
        }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return results


def run_rl_comparison(config: dict, output_dirs: dict) -> dict:
    """
    实验2: RL 基线对比
    """
    print("\n" + "="*60)
    print("Experiment 2: RL Baseline Comparison")
    print("="*60)
    
    # 加载数据
    loader = RealDataLoader()
    if config['data']['source'] == 'synthetic':
        from src.utils import load_market_data
        data = load_market_data(config['data']['path'])
    else:
        data = loader.load_caiso_data(
            config['data']['start_date'],
            config['data']['end_date']
        )
    
    n_episodes = config.get('rl_baselines', {}).get('n_episodes', 100)
    
    results = {}
    
    # 1. Q-Learning
    print("\nTraining Q-Learning...")
    q_agent = SimpleQAgent(
        learning_rate=0.1,
        epsilon_decay=0.995
    )
    
    for episode in range(n_episodes):
        env = BatteryEnv(data.copy())
        q_agent.train_episode(env)
        
        if (episode + 1) % 20 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}")
    
    eval_env = BatteryEnv(data.copy())
    q_reward, q_history = q_agent.evaluate(eval_env)
    results['q_learning'] = {
        'total_profit': q_reward,
        'training_history': q_agent.training_history
    }
    print(f"  Q-Learning Final Profit: ${q_reward:.2f}")
    
    # 2. DQN
    print("\nTraining DQN...")
    dqn_agent = DQNAgent(
        learning_rate=0.001,
        epsilon_decay=0.995
    )
    
    for episode in range(n_episodes):
        env = BatteryEnv(data.copy())
        dqn_agent.train_episode(env)
        
        if (episode + 1) % 20 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}")
    
    eval_env = BatteryEnv(data.copy())
    dqn_reward, dqn_history = dqn_agent.evaluate(eval_env)
    results['dqn'] = {
        'total_profit': dqn_reward,
        'training_history': dqn_agent.training_history
    }
    print(f"  DQN Final Profit: ${dqn_reward:.2f}")
    
    # 3. MPC (理想情况)
    print("\nEvaluating MPC (Perfect Forecast)...")
    mpc_agent = MPCBaseline(horizon=24)
    mpc_agent.set_price_forecast(data['price'].tolist())
    
    eval_env = BatteryEnv(data.copy())
    obs, _ = eval_env.reset(options={"initial_soc": 0.5})
    mpc_reward = 0
    
    while obs is not None:
        action = mpc_agent.decide(obs)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        mpc_reward += reward
        if done:
            break
    
    results['mpc'] = {'total_profit': mpc_reward}
    print(f"  MPC (Upper Bound) Profit: ${mpc_reward:.2f}")
    
    # 4. Rule-based (对比)
    print("\nEvaluating Rule-based...")
    rule_agent = RuleAgent()
    eval_env = BatteryEnv(data.copy())
    obs, _ = eval_env.reset(options={"initial_soc": 0.5})
    rule_reward = 0
    
    while obs is not None:
        action = rule_agent.decide(obs)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        rule_reward += reward
        if done:
            break
    
    results['rule'] = {'total_profit': rule_reward}
    print(f"  Rule-based Profit: ${rule_reward:.2f}")
    
    # 保存结果
    results_path = os.path.join(output_dirs['data'], 'rl_comparison_results.json')
    
    serializable_results = {}
    for name, res in results.items():
        serializable_results[name] = {
            'total_profit': res['total_profit'],
            'training_history': res.get('training_history', [])
        }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    return results


def run_ablation_study(config: dict, output_dirs: dict) -> dict:
    """
    实验3: 消融实验
    """
    print("\n" + "="*60)
    print("Experiment 3: Ablation Study")
    print("="*60)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("[Warning] OPENAI_API_KEY not set, skipping ablation study")
        return {}
    
    # 加载数据
    loader = RealDataLoader()
    if config['data']['source'] == 'synthetic':
        from src.utils import load_market_data
        data = load_market_data(config['data']['path'])
    else:
        data = loader.load_caiso_data(
            config['data']['start_date'],
            config['data']['end_date']
        )
    
    runner = ExperimentRunner(config=config, output_dir=output_dirs['logs'])
    
    # 运行消融实验
    ablation_results = runner.run_ablation_experiments(data)
    
    # 保存结果
    results_path = os.path.join(output_dirs['data'], 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)
    
    return ablation_results


def run_cross_market_evaluation(config: dict, output_dirs: dict) -> dict:
    """
    实验4: 跨市场评估
    """
    print("\n" + "="*60)
    print("Experiment 4: Cross-Market Evaluation")
    print("="*60)
    
    loader = RealDataLoader()
    markets = ['caiso', 'pjm', 'ercot']
    
    results = {}
    
    for market in markets:
        print(f"\nEvaluating on {market.upper()} market...")
        
        if market == 'caiso':
            data = loader.load_caiso_data(
                config['data']['start_date'],
                config['data']['end_date']
            )
        elif market == 'pjm':
            data = loader.load_pjm_data(
                config['data']['start_date'],
                config['data']['end_date']
            )
        else:  # ercot
            data = loader.load_ercot_data(
                config['data']['start_date'],
                config['data']['end_date']
            )
        
        market_results = {}
        
        # Rule-based
        rule_agent = RuleAgent()
        env = BatteryEnv(data.copy())
        obs, _ = env.reset(options={"initial_soc": 0.5})
        total = 0
        
        while obs is not None:
            action = rule_agent.decide(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
            if done:
                break
        
        market_results['rule'] = total
        print(f"  Rule-based: ${total:.2f}")
        
        # LLM agents (if API key available)
        if os.getenv('OPENAI_API_KEY'):
            # Zero-shot
            llm_agent = SimpleLLMAgent(model=config['agents']['llm']['model'])
            env = BatteryEnv(data.copy())
            obs, _ = env.reset(options={"initial_soc": 0.5})
            total = 0
            
            while obs is not None:
                action = llm_agent.decide(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total += reward
                if done:
                    break
            
            market_results['zero_shot'] = total
            print(f"  Zero-shot LLM: ${total:.2f}")
        
        results[market] = market_results
    
    # 保存结果
    results_path = os.path.join(output_dirs['data'], 'cross_market_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def generate_paper_figures(results: dict, output_dirs: dict, config: dict):
    """
    生成论文图表
    """
    print("\n" + "="*60)
    print("Generating Paper Figures")
    print("="*60)
    
    viz = AcademicVisualizer(
        style='academic',
        figsize=(10, 6),
        dpi=300
    )
    
    # 准备数据
    if 'main_experiment' in results:
        main_results = results['main_experiment']
        
        # 提取每日收益数据
        daily_returns = {}
        for agent, data in main_results.items():
            if 'daily_returns' in data and data['daily_returns']:
                daily_returns[agent] = data['daily_returns'][0]  # 取第一次运行
        
        if daily_returns:
            # 图1: 累积收益曲线
            print("  Generating cumulative profit plot...")
            fig1 = viz.plot_cumulative_profits_with_ci(
                {k: [v] for k, v in daily_returns.items()},
                title='Cumulative Daily Profits by Strategy'
            )
            viz.save_figure(fig1, output_dirs['figures'], 'fig1_cumulative_profits')
            
            # 图2: 每日收益箱线图
            print("  Generating daily profit boxplot...")
            fig2 = viz.plot_daily_boxplot(
                daily_returns,
                title='Daily Profit Distribution'
            )
            viz.save_figure(fig2, output_dirs['figures'], 'fig2_daily_boxplot')
    
    # 图3: RL 训练曲线 (如果有)
    if 'rl_comparison' in results:
        print("  Generating RL training curves...")
        rl_results = results['rl_comparison']
        
        import matplotlib.pyplot as plt
        fig3, ax = plt.subplots(figsize=(10, 6))
        
        for name, data in rl_results.items():
            if 'training_history' in data and data['training_history']:
                ax.plot(data['training_history'], label=name, alpha=0.8)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward ($)')
        ax.set_title('RL Agent Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        viz.save_figure(fig3, output_dirs['figures'], 'fig3_rl_training')
        plt.close(fig3)
    
    # 图4: 跨市场对比 (如果有)
    if 'cross_market' in results:
        print("  Generating cross-market comparison...")
        cross_results = results['cross_market']
        
        import matplotlib.pyplot as plt
        fig4, ax = plt.subplots(figsize=(10, 6))
        
        markets = list(cross_results.keys())
        agents = list(cross_results[markets[0]].keys())
        x = np.arange(len(markets))
        width = 0.35
        
        for i, agent in enumerate(agents):
            values = [cross_results[m].get(agent, 0) for m in markets]
            ax.bar(x + i * width, values, width, label=agent)
        
        ax.set_xlabel('Market')
        ax.set_ylabel('Total Profit ($)')
        ax.set_title('Cross-Market Performance Comparison')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([m.upper() for m in markets])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        viz.save_figure(fig4, output_dirs['figures'], 'fig4_cross_market')
        plt.close(fig4)
    
    print(f"\nFigures saved to {output_dirs['figures']}")


def generate_latex_tables(results: dict, output_dirs: dict):
    """
    生成 LaTeX 表格
    """
    print("\n" + "="*60)
    print("Generating LaTeX Tables")
    print("="*60)
    
    calc = MetricsCalculator()
    
    # 表1: 主实验结果
    if 'main_experiment' in results:
        main_results = results['main_experiment']
        
        table_data = []
        for agent, data in main_results.items():
            if 'summary' in data:
                summary = data['summary']
                table_data.append({
                    'Strategy': agent.replace('_', ' ').title(),
                    'Mean Profit ($)': f"{summary.get('mean_profit', 0):.2f}",
                    'Std Dev ($)': f"{summary.get('std_profit', 0):.2f}",
                    'Sharpe Ratio': f"{summary.get('sharpe_ratio', 0):.3f}",
                    'Max Drawdown ($)': f"{summary.get('max_drawdown', 0):.2f}"
                })
        
        if table_data:
            df = pd.DataFrame(table_data)
            
            latex_table = df.to_latex(
                index=False,
                caption='Main Experiment Results',
                label='tab:main_results',
                column_format='l' + 'r' * (len(df.columns) - 1)
            )
            
            table_path = os.path.join(output_dirs['tables'], 'table1_main_results.tex')
            with open(table_path, 'w') as f:
                f.write(latex_table)
            
            print(f"  Saved: {table_path}")
    
    # 表2: RL 对比结果
    if 'rl_comparison' in results:
        rl_results = results['rl_comparison']
        
        table_data = []
        for agent, data in rl_results.items():
            table_data.append({
                'Method': agent.replace('_', ' ').upper(),
                'Total Profit ($)': f"{data.get('total_profit', 0):.2f}"
            })
        
        if table_data:
            df = pd.DataFrame(table_data)
            
            latex_table = df.to_latex(
                index=False,
                caption='RL Baseline Comparison',
                label='tab:rl_comparison',
                column_format='lr'
            )
            
            table_path = os.path.join(output_dirs['tables'], 'table2_rl_comparison.tex')
            with open(table_path, 'w') as f:
                f.write(latex_table)
            
            print(f"  Saved: {table_path}")
    
    # 表3: 跨市场结果
    if 'cross_market' in results:
        cross_results = results['cross_market']
        
        table_data = []
        for market, data in cross_results.items():
            row = {'Market': market.upper()}
            for agent, profit in data.items():
                row[agent.replace('_', ' ').title()] = f"${profit:.2f}"
            table_data.append(row)
        
        if table_data:
            df = pd.DataFrame(table_data)
            
            latex_table = df.to_latex(
                index=False,
                caption='Cross-Market Performance',
                label='tab:cross_market',
                column_format='l' + 'r' * (len(df.columns) - 1)
            )
            
            table_path = os.path.join(output_dirs['tables'], 'table3_cross_market.tex')
            with open(table_path, 'w') as f:
                f.write(latex_table)
            
            print(f"  Saved: {table_path}")


def main():
    parser = argparse.ArgumentParser(description='Reproduce paper experiments')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='paper_results',
                       help='Output directory')
    parser.add_argument('--experiments', nargs='+', 
                       default=['main', 'rl', 'cross_market'],
                       choices=['main', 'rl', 'ablation', 'cross_market', 'all'],
                       help='Which experiments to run')
    parser.add_argument('--skip-figures', action='store_true',
                       help='Skip figure generation')
    parser.add_argument('--skip-tables', action='store_true',
                       help='Skip LaTeX table generation')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = os.path.join(args.output_dir, timestamp)
    output_dirs = setup_output_dirs(output_base)
    
    print("\n" + "="*60)
    print("Paper Results Reproduction Script")
    print("="*60)
    print(f"Output directory: {output_base}")
    print(f"Experiments: {args.experiments}")
    
    # 保存配置
    config_path = os.path.join(output_dirs['base'], 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    all_results = {}
    
    # 运行选定的实验
    experiments = args.experiments
    if 'all' in experiments:
        experiments = ['main', 'rl', 'ablation', 'cross_market']
    
    if 'main' in experiments:
        all_results['main_experiment'] = run_main_experiments(config, output_dirs)
    
    if 'rl' in experiments:
        all_results['rl_comparison'] = run_rl_comparison(config, output_dirs)
    
    if 'ablation' in experiments:
        all_results['ablation'] = run_ablation_study(config, output_dirs)
    
    if 'cross_market' in experiments:
        all_results['cross_market'] = run_cross_market_evaluation(config, output_dirs)
    
    # 生成图表和表格
    if not args.skip_figures:
        generate_paper_figures(all_results, output_dirs, config)
    
    if not args.skip_tables:
        generate_latex_tables(all_results, output_dirs)
    
    # 保存完整结果
    results_summary_path = os.path.join(output_dirs['base'], 'all_results_summary.json')
    
    # 简化结果用于保存
    summary = {}
    for exp_name, exp_results in all_results.items():
        if isinstance(exp_results, dict):
            summary[exp_name] = {
                k: {
                    'total_profit': v.get('total_profit') or v.get('summary', {}).get('mean_profit'),
                }
                for k, v in exp_results.items()
                if isinstance(v, dict)
            }
    
    with open(results_summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print(f"Results saved to: {output_base}")
    print("="*60)


if __name__ == '__main__':
    main()
