#!/usr/bin/env python3
"""
Agent-Generates-Agent (AGA) 训练脚本

这个脚本专门用于训练和评估 MetaReflexionAgent：
1. 运行多天模拟让 Meta-Agent 学习生成策略代码
2. 保存最佳策略代码到文件
3. 对比评估各种 Agent 的表现
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

from src.agents import MetaReflexionAgent, RuleAgent, create_agent
from src.env import BatteryEnv
from src.utils import load_market_data

# 加载环境变量
load_dotenv()


def run_aga_training(
    agent: MetaReflexionAgent,
    env: BatteryEnv,
    num_days: int = 14,
    verbose: bool = True
) -> Dict:
    """
    运行 AGA 训练循环
    
    Args:
        agent: MetaReflexionAgent 实例
        env: 电池环境
        num_days: 训练天数
        verbose: 是否打印详细信息
    
    Returns:
        训练结果统计
    """
    agent.reset()
    all_records = []
    daily_profits = []
    
    for day in range(num_days):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Day {day + 1}/{num_days}")
            print(f"{'='*60}")
        
        # 运行一天
        obs, _ = env.reset()
        daily_records = []
        
        for hour in range(24):
            action = agent.decide(obs)
            
            next_obs, reward, done, truncated, info = env.step(action)
            
            # 记录
            record = {
                'day': day,
                'hour': hour,
                'price': obs['price'],
                'soc': obs['soc'],
                'action': action,
                'reward': reward,
                'cumulative': info.get('cumulative_profit', 0)
            }
            daily_records.append(record)
            all_records.append(record)
            agent.record_transaction(record)
            
            obs = next_obs
        
        # 计算当天收益
        daily_profit = sum(r['reward'] for r in daily_records)
        daily_profits.append(daily_profit)
        
        if verbose:
            print(f"  Daily Profit: ${daily_profit:.4f}")
            
            # 显示动作分布
            actions = [r['action'] for r in daily_records]
            print(f"  Actions: CHARGE={actions.count('CHARGE')}, "
                  f"DISCHARGE={actions.count('DISCHARGE')}, "
                  f"HOLD={actions.count('HOLD')}")
        
        # 每日反思和策略更新
        agent.end_of_day(daily_records)
    
    # 计算总体统计
    total_profit = sum(daily_profits)
    avg_profit = total_profit / num_days
    
    results = {
        'total_profit': total_profit,
        'avg_daily_profit': avg_profit,
        'daily_profits': daily_profits,
        'best_profit': agent.best_profit,
        'num_strategies_tried': len(agent.code_history),
        'total_llm_calls': agent.total_llm_calls,
        'all_records': all_records
    }
    
    return results


def save_best_strategy(agent: MetaReflexionAgent, output_dir: str = "outputs") -> str:
    """
    保存最佳策略代码到文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"best_strategy_{timestamp}.py"
    filepath = os.path.join(output_dir, filename)
    
    code = agent.get_best_strategy_code()
    
    with open(filepath, 'w') as f:
        f.write(f'"""\nAuto-generated Strategy by MetaReflexionAgent\n')
        f.write(f'Generated at: {datetime.now().isoformat()}\n')
        f.write(f'Best single-day profit: ${agent.best_profit:.4f}\n')
        f.write(f'"""\n\n')
        f.write(code)
    
    print(f"\nBest strategy saved to: {filepath}")
    return filepath


def compare_agents(
    env: BatteryEnv,
    meta_agent: MetaReflexionAgent,
    num_eval_days: int = 7,
    verbose: bool = True
) -> Dict:
    """
    对比评估 Meta-Agent 生成的策略与其他 Agent
    """
    results = {}
    
    # 1. Rule-based Agent
    if verbose:
        print("\n" + "="*60)
        print("Evaluating Rule-based Agent")
        print("="*60)
    
    rule_agent = RuleAgent()
    rule_profits = []
    
    for day in range(num_eval_days):
        obs, _ = env.reset()
        daily_profit = 0
        for _ in range(24):
            action = rule_agent.decide(obs)
            obs, reward, _, _, _ = env.step(action)
            daily_profit += reward
        rule_profits.append(daily_profit)
    
    results['rule'] = {
        'total': sum(rule_profits),
        'avg': sum(rule_profits) / num_eval_days,
        'daily': rule_profits
    }
    
    if verbose:
        print(f"  Total: ${results['rule']['total']:.4f}, Avg: ${results['rule']['avg']:.4f}")
    
    # 2. Meta-Agent 生成的策略 (使用固定的最佳策略)
    if verbose:
        print("\n" + "="*60)
        print("Evaluating Meta-Agent's Best Strategy")
        print("="*60)
    
    # 加载最佳策略
    if meta_agent.best_code:
        from src.code_executor import StrategyLoader
        loader = StrategyLoader()
        best_strategy, error = loader.load_strategy(meta_agent.best_code)
        
        if best_strategy:
            meta_profits = []
            for day in range(num_eval_days):
                obs, _ = env.reset()
                daily_profit = 0
                for _ in range(24):
                    try:
                        action = best_strategy.decide(obs)
                        action = action.upper().strip() if isinstance(action, str) else "HOLD"
                        if action not in ["CHARGE", "DISCHARGE", "HOLD"]:
                            action = "HOLD"
                    except:
                        action = "HOLD"
                    
                    obs, reward, _, _, _ = env.step(action)
                    daily_profit += reward
                meta_profits.append(daily_profit)
            
            results['meta_best'] = {
                'total': sum(meta_profits),
                'avg': sum(meta_profits) / num_eval_days,
                'daily': meta_profits
            }
            
            if verbose:
                print(f"  Total: ${results['meta_best']['total']:.4f}, "
                      f"Avg: ${results['meta_best']['avg']:.4f}")
        else:
            if verbose:
                print(f"  Failed to load best strategy: {error}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="AGA Training Script")
    parser.add_argument("--train-days", type=int, default=14,
                       help="Number of training days")
    parser.add_argument("--eval-days", type=int, default=7,
                       help="Number of evaluation days")
    parser.add_argument("--model", type=str, default=None,
                       help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="LLM temperature")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory for results")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    parser.add_argument("--save-strategy", action="store_true",
                       help="Save best strategy to file")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with other agents after training")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    print("\n" + "="*60)
    print("Agent-Generates-Agent (AGA) Training")
    print("="*60)
    
    # 加载数据
    if verbose:
        print("\nLoading market data...")
    
    df = load_market_data()
    
    # 创建环境
    env = BatteryEnv(df)
    
    # 创建 Meta-Agent
    if verbose:
        print("\nInitializing MetaReflexionAgent...")
    
    agent_kwargs = {"temperature": args.temperature}
    if args.model:
        agent_kwargs["model_name"] = args.model
    
    meta_agent = MetaReflexionAgent(**agent_kwargs)
    
    # 训练
    if verbose:
        print(f"\nStarting {args.train_days}-day training...")
    
    results = run_aga_training(
        agent=meta_agent,
        env=env,
        num_days=args.train_days,
        verbose=verbose
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("Training Results Summary")
    print("="*60)
    print(f"Total Profit: ${results['total_profit']:.4f}")
    print(f"Average Daily Profit: ${results['avg_daily_profit']:.4f}")
    print(f"Best Single-Day Profit: ${results['best_profit']:.4f}")
    print(f"Strategies Generated: {results['num_strategies_tried']}")
    print(f"Total LLM Calls: {results['total_llm_calls']}")
    
    # 打印每日收益
    print("\nDaily Profits:")
    for i, profit in enumerate(results['daily_profits']):
        status = "✓" if profit > 0 else "✗"
        print(f"  Day {i+1}: ${profit:+.4f} {status}")
    
    # 保存最佳策略
    if args.save_strategy:
        save_best_strategy(meta_agent, args.output_dir)
    
    # 对比评估
    if args.compare:
        print("\n" + "="*60)
        print("Comparative Evaluation")
        print("="*60)
        
        compare_results = compare_agents(
            env=env,
            meta_agent=meta_agent,
            num_eval_days=args.eval_days,
            verbose=verbose
        )
        
        # 打印对比表
        print("\n" + "-"*40)
        print("Agent Performance Comparison")
        print("-"*40)
        print(f"{'Agent':<20} {'Total':>12} {'Avg/Day':>12}")
        print("-"*40)
        
        for agent_name, data in compare_results.items():
            print(f"{agent_name:<20} ${data['total']:>10.4f} ${data['avg']:>10.4f}")
    
    # 保存完整结果
    output_path = os.path.join(args.output_dir, "aga_training_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 转换为可序列化格式
    save_results = {
        'training': {
            'total_profit': results['total_profit'],
            'avg_daily_profit': results['avg_daily_profit'],
            'best_profit': results['best_profit'],
            'num_strategies': results['num_strategies_tried'],
            'llm_calls': results['total_llm_calls'],
            'daily_profits': results['daily_profits']
        },
        'best_code': meta_agent.get_best_strategy_code(),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    if verbose:
        print(f"\nResults saved to: {output_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
