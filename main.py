"""
LLM-Based Battery Arbitrage Agent with Reflexion
主程序入口 - 运行实验循环并对比不同 Agent 的性能

Usage:
    uv run python main.py
    uv run python main.py --days 7
    uv run python main.py --agents rule reflexion
"""
import os
import argparse
from typing import Dict, List
from dotenv import load_dotenv

from src.env import BatteryEnv
from src.agents import RuleAgent, SimpleLLMAgent, ReflexionAgent, BaseAgent
from src.utils import (
    load_market_data,
    plot_cumulative_profits,
    plot_daily_profits,
    plot_soc_profile,
    plot_action_distribution,
    print_experiment_summary,
    create_results_dataframe,
)

# 加载环境变量
load_dotenv()


def run_single_agent(
    agent: BaseAgent,
    env: BatteryEnv,
    num_days: int,
    verbose: bool = True
) -> Dict:
    """
    运行单个 Agent 的实验
    
    Args:
        agent: Agent 实例
        env: 环境实例
        num_days: 运行天数
        verbose: 是否打印详细信息
        
    Returns:
        包含结果指标的字典
    """
    # 重置
    agent.reset()
    obs = env.reset()
    
    hourly_profits = []
    daily_profits = []
    all_history = []
    
    for day in range(num_days):
        daily_buffer = []
        daily_profit = 0
        
        for hour in range(24):
            if obs is None:
                break
            
            # Agent 决策
            action = agent.decide(obs)
            
            # 环境执行
            next_obs, reward, done, info = env.step(action)
            
            # 记录
            hourly_profits.append(reward)
            daily_profit += reward
            daily_buffer.append(info)
            all_history.append(info)
            
            # 如果是 ReflexionAgent，记录交易
            if hasattr(agent, 'record_transaction'):
                agent.record_transaction(info)
            
            obs = next_obs
            
            if done:
                break
        
        # 每日结束处理
        daily_profits.append(daily_profit)
        
        # 反思（如果 Agent 支持）
        reflection = agent.end_of_day(daily_buffer)
        
        if verbose:
            print(f"  Day {day + 1}: Profit ${daily_profit:.4f}", end="")
            if reflection and hasattr(agent, 'get_memory_summary'):
                # 只打印最新的反思摘要
                print(f" | Reflection: {reflection[:80]}..." if len(str(reflection)) > 80 else f" | {reflection[:80]}")
            else:
                print()
    
    # 汇总统计
    total_profit = sum(hourly_profits)
    total_cost = sum(h['grid_cost'] for h in all_history)
    total_revenue = sum(h['grid_revenue'] for h in all_history)
    charge_count = sum(1 for h in all_history if h['action'] == 'CHARGE')
    discharge_count = sum(1 for h in all_history if h['action'] == 'DISCHARGE')
    hold_count = len(all_history) - charge_count - discharge_count
    
    return {
        'agent_name': agent.name,
        'total_profit': total_profit,
        'total_cost': total_cost,
        'total_revenue': total_revenue,
        'hourly_profits': hourly_profits,
        'daily_profits': daily_profits,
        'history': all_history,
        'llm_calls': getattr(agent, 'total_llm_calls', 0),
        'charge_count': charge_count,
        'discharge_count': discharge_count,
        'hold_count': hold_count,
        'final_memory': agent.get_memory_summary() if hasattr(agent, 'get_memory_summary') else None
    }


def run_experiment(
    agents: List[BaseAgent],
    df,
    num_days: int = 7,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    运行完整实验，对比多个 Agent
    
    Args:
        agents: Agent 列表
        df: 市场数据 DataFrame
        num_days: 模拟天数
        verbose: 是否打印详细信息
        
    Returns:
        {agent_name: results} 字典
    """
    results = {}
    
    for agent in agents:
        print(f"\n{'='*50}")
        print(f"🤖 Running: {agent.name}")
        print('='*50)
        
        # 为每个 Agent 创建新的环境实例
        env = BatteryEnv(df)
        
        result = run_single_agent(agent, env, num_days, verbose)
        results[agent.name] = result
        
        print(f"\n✅ {agent.name} completed: Total Profit = ${result['total_profit']:.4f}")
        if result['llm_calls'] > 0:
            print(f"   LLM API calls: {result['llm_calls']}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Battery Arbitrage Agent Experiment')
    parser.add_argument('--days', type=int, default=7, help='Number of days to simulate')
    parser.add_argument('--agents', nargs='+', default=['rule', 'reflexion'],
                        choices=['rule', 'simple_llm', 'reflexion'],
                        help='Agents to run')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='LLM model to use')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print detailed output')
    
    args = parser.parse_args()
    
    # 检查 API Key
    if 'simple_llm' in args.agents or 'reflexion' in args.agents:
        if not os.getenv('OPENAI_API_KEY'):
            print("⚠️  Warning: OPENAI_API_KEY not found in environment.")
            print("   Please set it in .env file or export it.")
            print("   Running only rule-based agent...\n")
            args.agents = ['rule']
    
    print("🔋 Battery Arbitrage Agent Experiment")
    print("="*50)
    print(f"📅 Simulation period: {args.days} days")
    print(f"🤖 Agents: {', '.join(args.agents)}")
    print(f"🧠 Model: {args.model}")
    print("="*50)
    
    # 加载数据
    print("\n📊 Loading market data...")
    df = load_market_data()
    print(f"   Loaded {len(df)} hours of data")
    print(f"   Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    # 创建 Agent
    agents = []
    
    if 'rule' in args.agents:
        agents.append(RuleAgent())
    
    if 'simple_llm' in args.agents:
        agents.append(SimpleLLMAgent(model_name=args.model))
    
    if 'reflexion' in args.agents:
        agents.append(ReflexionAgent(model_name=args.model))
    
    # 运行实验
    results = run_experiment(agents, df, num_days=args.days, verbose=args.verbose)
    
    # 打印摘要
    print_experiment_summary(results)
    
    # 保存结果到 CSV
    results_df = create_results_dataframe(results)
    results_df.to_csv('experiment_results.csv', index=False)
    print("💾 Results saved to experiment_results.csv")
    
    # 绘图
    if not args.no_plot:
        print("\n📈 Generating plots...")
        
        # 累积利润对比
        hourly_results = {name: r['hourly_profits'] for name, r in results.items()}
        plot_cumulative_profits(
            hourly_results,
            title=f"Cumulative Profit Comparison ({args.days} Days)",
            save_path="cumulative_profits.png"
        )
        
        # 每日利润对比
        daily_results = {name: r['daily_profits'] for name, r in results.items()}
        plot_daily_profits(
            daily_results,
            title=f"Daily Profit Comparison",
            save_path="daily_profits.png"
        )
        
        # 动作分布
        action_results = {
            name: {
                'CHARGE': r['charge_count'],
                'DISCHARGE': r['discharge_count'],
                'HOLD': r['hold_count']
            }
            for name, r in results.items()
        }
        plot_action_distribution(
            action_results,
            title="Action Distribution by Agent",
            save_path="action_distribution.png"
        )
        
        # SOC 曲线（只画第一个 Agent）
        first_agent = list(results.keys())[0]
        plot_soc_profile(
            results[first_agent]['history'],
            df,
            title=f"Battery SOC Profile ({first_agent})",
            save_path=f"soc_profile_{first_agent}.png"
        )
    
    # 打印 ReflexionAgent 的最终记忆
    for name, result in results.items():
        if result.get('final_memory'):
            print(f"\n📝 {name}'s Final Strategy Memory:")
            print("-" * 40)
            print(result['final_memory'])
            print("-" * 40)
    
    return results


if __name__ == "__main__":
    main()
