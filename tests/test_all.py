"""
单元测试模块
确保代码质量和可复现性
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import BatteryEnv
from src.agents import RuleAgent, SimpleLLMAgent, ReflexionAgent
from src.metrics import MetricsCalculator
from src.data_loader import RealDataLoader


class TestBatteryEnv(unittest.TestCase):
    """测试电池环境"""
    
    def setUp(self):
        """创建测试数据"""
        # 3天数据 = 72小时
        prices = []
        for _ in range(3):  # 3天
            # 每天24小时的价格模式
            day_prices = [0.1]*8 + [0.2]*8 + [0.5]*4 + [0.2]*4  # 24小时
            prices.extend(day_prices)
        
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=72, freq='h'),
            'price': prices,
            'load': [2.0] * 72
        })
        self.env = BatteryEnv(self.df)
    
    def test_initialization(self):
        """测试环境初始化"""
        self.assertEqual(self.env.capacity, 13.5)
        self.assertEqual(self.env.max_power, 5.0)
        self.assertEqual(self.env.efficiency, 0.9)
        self.assertEqual(self.env.current_soc, 0.5)
    
    def test_reset(self):
        """测试重置功能"""
        obs = self.env.reset(initial_soc=0.3)
        self.assertEqual(obs['soc'], 30.0)
        self.assertEqual(self.env.current_step, 0)
    
    def test_charge_action(self):
        """测试充电动作"""
        self.env.reset(initial_soc=0.5)
        initial_soc = self.env.current_soc
        
        obs, reward, done, info = self.env.step("CHARGE")
        
        # SOC 应该增加
        self.assertGreater(self.env.current_soc, initial_soc)
        # 充电应该产生成本（负奖励）
        self.assertLess(reward, 0)
        self.assertEqual(info['action'], 'CHARGE')
    
    def test_discharge_action(self):
        """测试放电动作"""
        self.env.reset(initial_soc=0.8)
        initial_soc = self.env.current_soc
        
        obs, reward, done, info = self.env.step("DISCHARGE")
        
        # SOC 应该减少
        self.assertLess(self.env.current_soc, initial_soc)
        # 放电应该产生收益（正奖励）
        self.assertGreater(reward, 0)
        self.assertEqual(info['action'], 'DISCHARGE')
    
    def test_hold_action(self):
        """测试保持动作"""
        self.env.reset(initial_soc=0.5)
        initial_soc = self.env.current_soc
        
        obs, reward, done, info = self.env.step("HOLD")
        
        # SOC 应该不变
        self.assertEqual(self.env.current_soc, initial_soc)
        # 无操作应该无奖励
        self.assertEqual(reward, 0)
    
    def test_soc_limits(self):
        """测试 SOC 边界限制"""
        # 测试不能过充
        self.env.reset(initial_soc=0.95)
        self.env.step("CHARGE")
        self.assertLessEqual(self.env.current_soc, 1.0)
        
        # 测试不能过放
        self.env.reset(initial_soc=0.15)
        self.env.step("DISCHARGE")
        self.assertGreaterEqual(self.env.current_soc, 0.1)
    
    def test_episode_completion(self):
        """测试完整回合"""
        self.env.reset()
        done = False
        steps = 0
        
        while not done:
            _, _, done, _ = self.env.step("HOLD")
            steps += 1
        
        self.assertEqual(steps, len(self.df))


class TestRuleAgent(unittest.TestCase):
    """测试规则基线 Agent"""
    
    def test_charge_decision(self):
        """测试低价充电决策"""
        agent = RuleAgent()
        
        obs = {'price': 0.10, 'soc': 50, 'hour': 3}
        action = agent.decide(obs)
        self.assertEqual(action, 'CHARGE')
    
    def test_discharge_decision(self):
        """测试高价放电决策"""
        agent = RuleAgent()
        
        obs = {'price': 0.50, 'soc': 50, 'hour': 18}
        action = agent.decide(obs)
        self.assertEqual(action, 'DISCHARGE')
    
    def test_hold_decision(self):
        """测试中等价格保持决策"""
        agent = RuleAgent()
        
        obs = {'price': 0.25, 'soc': 50, 'hour': 12}
        action = agent.decide(obs)
        self.assertEqual(action, 'HOLD')
    
    def test_soc_constraints(self):
        """测试 SOC 约束"""
        agent = RuleAgent()
        
        # 电量满时不充电
        obs = {'price': 0.10, 'soc': 95, 'hour': 3}
        action = agent.decide(obs)
        self.assertEqual(action, 'HOLD')
        
        # 电量低时不放电
        obs = {'price': 0.50, 'soc': 5, 'hour': 18}
        action = agent.decide(obs)
        self.assertEqual(action, 'HOLD')


class TestMetricsCalculator(unittest.TestCase):
    """测试指标计算器"""
    
    def setUp(self):
        self.calc = MetricsCalculator()
    
    def test_sharpe_ratio(self):
        """测试夏普比率计算"""
        # 稳定正收益
        returns = [0.01] * 100
        sharpe = self.calc.calculate_sharpe_ratio(returns, annualize=False)
        self.assertGreater(sharpe, 0)
        
        # 零收益
        returns = [0] * 100
        sharpe = self.calc.calculate_sharpe_ratio(returns)
        self.assertEqual(sharpe, 0)
    
    def test_max_drawdown(self):
        """测试最大回撤计算"""
        # 先涨后跌
        cumulative = [0, 1, 2, 3, 4, 5, 4, 3, 2, 3]
        max_dd, start, end = self.calc.calculate_max_drawdown(cumulative)
        
        self.assertEqual(max_dd, 3)  # 从5跌到2
        self.assertEqual(start, 5)  # 高点索引
        self.assertEqual(end, 8)    # 低点索引
    
    def test_profit_factor(self):
        """测试盈亏比计算"""
        returns = [1, 1, 1, -0.5, -0.5]
        pf = self.calc.calculate_profit_factor(returns)
        
        self.assertEqual(pf, 3.0)  # 3/1
    
    def test_win_rate(self):
        """测试胜率计算"""
        returns = [1, 1, 1, -1, 0]
        win_rate = self.calc.calculate_win_rate(returns)
        
        self.assertEqual(win_rate, 0.6)  # 3/5
    
    def test_paired_t_test(self):
        """测试配对 t 检验"""
        np.random.seed(42)
        
        # A 明显优于 B
        returns_a = np.random.normal(1, 0.1, 30)
        returns_b = np.random.normal(0.5, 0.1, 30)
        
        result = self.calc.paired_t_test(returns_a.tolist(), returns_b.tolist())
        
        self.assertLess(result['p_value'], 0.05)  # 应该显著
        self.assertGreater(result['cohens_d'], 0)  # A > B
    
    def test_bootstrap_ci(self):
        """测试 Bootstrap 置信区间"""
        np.random.seed(42)
        returns = np.random.normal(1, 0.5, 100).tolist()
        
        mean, lower, upper = self.calc.bootstrap_confidence_interval(returns)
        
        self.assertLess(lower, mean)
        self.assertGreater(upper, mean)
        self.assertAlmostEqual(mean, 1.0, delta=0.2)


class TestDataLoader(unittest.TestCase):
    """测试数据加载器"""
    
    def setUp(self):
        self.loader = RealDataLoader(cache_dir="test_cache")
    
    def test_caiso_data_generation(self):
        """测试 CAISO 风格数据生成"""
        df = self.loader.load_caiso_data('2024-01-01', '2024-01-03')
        
        self.assertIn('timestamp', df.columns)
        self.assertIn('price', df.columns)
        self.assertIn('load', df.columns)
        self.assertEqual(len(df), 48)  # 2天 = 48小时
    
    def test_price_spikes(self):
        """测试价格尖峰添加"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
            'price': [0.1] * 100,
            'load': [2.0] * 100
        })
        
        np.random.seed(42)
        df_with_spikes = self.loader.add_price_spikes(df, spike_probability=0.2)
        
        # 应该有一些价格尖峰
        self.assertTrue((df_with_spikes['price'] > 0.1).any())
    
    def tearDown(self):
        """清理测试缓存"""
        import shutil
        if os.path.exists("test_cache"):
            shutil.rmtree("test_cache")


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_simulation(self):
        """测试完整模拟流程"""
        # 创建数据
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=48, freq='h'),
            'price': [0.1]*8 + [0.2]*8 + [0.5]*4 + [0.2]*4 + 
                     [0.1]*8 + [0.2]*8 + [0.5]*4 + [0.2]*4,
            'load': [2.0] * 48
        })
        
        # 创建环境和 Agent
        env = BatteryEnv(df)
        agent = RuleAgent()
        
        # 运行模拟
        obs = env.reset()
        total_reward = 0
        
        while obs is not None:
            action = agent.decide(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        # 检查结果
        self.assertIsInstance(total_reward, float)
        self.assertEqual(len(env.history), 48)
    
    def test_reproducibility(self):
        """测试可复现性"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='h'),
            'price': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                      0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                      0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.1, 0.1],
            'load': [2.0] * 24
        })
        
        results = []
        for _ in range(3):
            env = BatteryEnv(df)
            agent = RuleAgent()
            
            obs = env.reset(initial_soc=0.5)
            total = 0
            
            while obs is not None:
                action = agent.decide(obs)
                obs, reward, done, _ = env.step(action)
                total += reward
                if done:
                    break
            
            results.append(total)
        
        # RuleAgent 是确定性的，结果应该相同
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestBatteryEnv))
    suite.addTests(loader.loadTestsFromTestCase(TestRuleAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
