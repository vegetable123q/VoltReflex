"""
强化学习基线对比模块
实现 DQN 和简单 RL Agent 作为对比基线
"""
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import deque
import random


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class SimpleQAgent:
    """
    简单 Q-Learning Agent (表格方法)
    作为基础强化学习对比
    """
    
    def __init__(
        self, 
        n_price_bins: int = 10,
        n_soc_bins: int = 10,
        n_hour_bins: int = 24,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.n_price_bins = n_price_bins
        self.n_soc_bins = n_soc_bins
        self.n_hour_bins = n_hour_bins
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q表: (price_bin, soc_bin, hour) -> [HOLD, CHARGE, DISCHARGE]
        self.q_table = np.zeros((n_price_bins, n_soc_bins, n_hour_bins, 3))
        
        self.actions = ['HOLD', 'CHARGE', 'DISCHARGE']
        
        # 价格区间 (假设价格在 0-1 范围)
        self.price_bins = np.linspace(0, 1, n_price_bins + 1)
        # SOC 区间 (0-100%)
        self.soc_bins = np.linspace(0, 100, n_soc_bins + 1)
        
        self.training_history = []
    
    def _discretize_state(self, obs: Dict) -> Tuple[int, int, int]:
        """将连续状态离散化"""
        price_bin = min(
            np.digitize(obs['price'], self.price_bins) - 1, 
            self.n_price_bins - 1
        )
        soc_bin = min(
            np.digitize(obs['soc'], self.soc_bins) - 1, 
            self.n_soc_bins - 1
        )
        hour = int(obs['hour']) % 24
        
        return max(0, price_bin), max(0, soc_bin), hour
    
    def decide(self, obs: Dict) -> str:
        """选择动作（带探索）"""
        state = self._discretize_state(obs)
        
        if random.random() < self.epsilon:
            # 探索：随机选择
            action_idx = random.randint(0, 2)
        else:
            # 利用：选择最优动作
            action_idx = np.argmax(self.q_table[state])
        
        # 安全检查
        if obs['soc'] >= 90 and action_idx == 1:  # 满电不充
            action_idx = 0
        if obs['soc'] <= 20 and action_idx == 2:  # 低电不放
            action_idx = 0
        
        return self.actions[action_idx]
    
    def update(self, obs: Dict, action: str, reward: float, 
               next_obs: Dict, done: bool):
        """更新 Q 值"""
        state = self._discretize_state(obs)
        next_state = self._discretize_state(next_obs) if next_obs else state
        action_idx = self.actions.index(action)
        
        # Q-Learning 更新
        current_q = self.q_table[state][action_idx]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state][action_idx] += self.lr * (target - current_q)
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_episode(self, env, max_steps: int = 1000) -> float:
        """训练一个回合"""
        obs, _ = env.reset(options={"initial_soc": 0.5})
        total_reward = 0
        step = 0
        
        while obs is not None and step < max_steps:
            action = self.decide(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            self.update(obs, action, reward, next_obs, done)
            
            total_reward += reward
            obs = next_obs
            step += 1
            
            if done:
                break
        
        self.decay_epsilon()
        self.training_history.append(total_reward)
        
        return total_reward
    
    def evaluate(self, env) -> Tuple[float, List[Dict]]:
        """评估模式（无探索）"""
        old_epsilon = self.epsilon
        self.epsilon = 0
        
        obs, _ = env.reset(options={"initial_soc": 0.5})
        total_reward = 0
        history = []
        
        while obs is not None:
            action = self.decide(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            history.append({**obs, 'action': action, 'reward': reward})
            total_reward += reward
            obs = next_obs
            
            if done:
                break
        
        self.epsilon = old_epsilon
        return total_reward, history


class DQNAgent:
    """
    Deep Q-Network Agent
    使用简单的神经网络近似 Q 函数
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 32,
        target_update_freq: int = 100
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.actions = ['HOLD', 'CHARGE', 'DISCHARGE']
        self.n_actions = 3
        
        # 初始化网络权重
        self._init_network()
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        self.training_steps = 0
        self.training_history = []
    
    def _init_network(self):
        """初始化神经网络权重（简单的双层 MLP）"""
        # 主网络
        self.W1 = np.random.randn(self.state_dim, self.hidden_dim) * 0.1
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
        self.b2 = np.zeros(self.hidden_dim)
        self.W3 = np.random.randn(self.hidden_dim, self.n_actions) * 0.1
        self.b3 = np.zeros(self.n_actions)
        
        # 目标网络
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
        self.target_W3 = self.W3.copy()
        self.target_b3 = self.b3.copy()
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _forward(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        """前向传播"""
        if use_target:
            W1, b1 = self.target_W1, self.target_b1
            W2, b2 = self.target_W2, self.target_b2
            W3, b3 = self.target_W3, self.target_b3
        else:
            W1, b1 = self.W1, self.b1
            W2, b2 = self.W2, self.b2
            W3, b3 = self.W3, self.b3
        
        h1 = self._relu(np.dot(state, W1) + b1)
        h2 = self._relu(np.dot(h1, W2) + b2)
        q_values = np.dot(h2, W3) + b3
        
        return q_values
    
    def _obs_to_state(self, obs: Dict) -> np.ndarray:
        """将观察转换为状态向量"""
        return np.array([
            obs['price'],
            obs['soc'] / 100.0,
            np.sin(2 * np.pi * obs['hour'] / 24),
            np.cos(2 * np.pi * obs['hour'] / 24)
        ])
    
    def decide(self, obs: Dict) -> str:
        """选择动作"""
        state = self._obs_to_state(obs)
        
        if random.random() < self.epsilon:
            action_idx = random.randint(0, 2)
        else:
            q_values = self._forward(state)
            action_idx = np.argmax(q_values)
        
        # 安全检查
        if obs['soc'] >= 90 and action_idx == 1:
            action_idx = 0
        if obs['soc'] <= 20 and action_idx == 2:
            action_idx = 0
        
        return self.actions[action_idx]
    
    def store_transition(self, obs: Dict, action: str, reward: float,
                        next_obs: Dict, done: bool):
        """存储经验"""
        state = self._obs_to_state(obs)
        next_state = self._obs_to_state(next_obs) if next_obs else np.zeros(self.state_dim)
        action_idx = self.actions.index(action)
        
        self.replay_buffer.push(state, action_idx, reward, next_state, done)
    
    def update(self):
        """更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 采样批次
        batch = self.replay_buffer.sample(self.batch_size)
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        
        # 计算目标 Q 值
        current_q = self._forward(states)
        next_q = self._forward(next_states, use_target=True)
        
        targets = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # 简单的梯度下降更新
        self._gradient_update(states, targets)
        
        self.training_steps += 1
        
        # 更新目标网络
        if self.training_steps % self.target_update_freq == 0:
            self._update_target_network()
    
    def _gradient_update(self, states: np.ndarray, targets: np.ndarray):
        """梯度更新（简化版）"""
        # 前向传播保存中间结果
        h1 = self._relu(np.dot(states, self.W1) + self.b1)
        h2 = self._relu(np.dot(h1, self.W2) + self.b2)
        output = np.dot(h2, self.W3) + self.b3
        
        # 计算误差
        error = output - targets
        
        # 反向传播
        d3 = error / self.batch_size
        dW3 = np.dot(h2.T, d3)
        db3 = np.sum(d3, axis=0)
        
        d2 = np.dot(d3, self.W3.T) * (h2 > 0)
        dW2 = np.dot(h1.T, d2)
        db2 = np.sum(d2, axis=0)
        
        d1 = np.dot(d2, self.W2.T) * (h1 > 0)
        dW1 = np.dot(states.T, d1)
        db1 = np.sum(d1, axis=0)
        
        # 更新权重
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def _update_target_network(self):
        """更新目标网络"""
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
        self.target_W3 = self.W3.copy()
        self.target_b3 = self.b3.copy()
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_episode(self, env, max_steps: int = 1000) -> float:
        """训练一个回合"""
        obs, _ = env.reset(options={"initial_soc": 0.5})
        total_reward = 0
        step = 0
        
        while obs is not None and step < max_steps:
            action = self.decide(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            self.store_transition(obs, action, reward, next_obs, done)
            self.update()
            
            total_reward += reward
            obs = next_obs
            step += 1
            
            if done:
                break
        
        self.decay_epsilon()
        self.training_history.append(total_reward)
        
        return total_reward
    
    def evaluate(self, env) -> Tuple[float, List[Dict]]:
        """评估模式"""
        old_epsilon = self.epsilon
        self.epsilon = 0
        
        obs, _ = env.reset(options={"initial_soc": 0.5})
        total_reward = 0
        history = []
        
        while obs is not None:
            action = self.decide(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            history.append({**obs, 'action': action, 'reward': reward})
            total_reward += reward
            obs = next_obs
            
            if done:
                break
        
        self.epsilon = old_epsilon
        return total_reward, history


class MPCBaseline:
    """
    Model Predictive Control 基线
    假设未来价格完美已知的理想情况
    """
    
    def __init__(self, horizon: int = 24):
        self.horizon = horizon
        self.future_prices = None
        self.current_index = 0
    
    def set_price_forecast(self, prices: List[float]):
        """设置价格预测（或真实未来价格）"""
        self.future_prices = prices
        self.current_index = 0
    
    def decide(self, obs: Dict) -> str:
        """基于未来价格信息做出最优决策"""
        if self.future_prices is None:
            return 'HOLD'
        
        current_price = obs['price']
        soc = obs['soc']
        
        # 获取未来价格窗口
        future_end = min(self.current_index + self.horizon, len(self.future_prices))
        future_window = self.future_prices[self.current_index:future_end]
        
        if len(future_window) == 0:
            self.current_index += 1
            return 'HOLD'
        
        max_future = max(future_window)
        min_future = min(future_window)
        
        # MPC 策略
        # 如果当前价格是未来窗口最低，充电
        # 如果当前价格是未来窗口最高，放电
        
        if current_price <= min_future * 1.1 and soc < 90:
            action = 'CHARGE'
        elif current_price >= max_future * 0.9 and soc > 20:
            action = 'DISCHARGE'
        else:
            action = 'HOLD'
        
        self.current_index += 1
        return action
    
    def reset(self):
        """重置"""
        self.current_index = 0


def train_rl_agent(agent, env_class, data, n_episodes: int = 100, 
                  verbose: bool = True) -> Dict:
    """
    训练 RL Agent
    
    Args:
        agent: RL Agent 实例
        env_class: 环境类
        data: 训练数据
        n_episodes: 训练回合数
        verbose: 是否打印进度
    
    Returns:
        训练结果字典
    """
    training_rewards = []
    
    for episode in range(n_episodes):
        env = env_class(data.copy())
        reward = agent.train_episode(env)
        training_rewards.append(reward)
        
        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(training_rewards[-10:])
            print(f"Episode {episode + 1}/{n_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return {
        'training_rewards': training_rewards,
        'final_epsilon': agent.epsilon,
        'total_episodes': n_episodes
    }


def compare_rl_baselines(data, n_episodes: int = 100) -> Dict:
    """
    比较不同 RL 基线的性能
    
    Args:
        data: 数据 DataFrame
        n_episodes: 训练回合数
    
    Returns:
        比较结果
    """
    from src.env import BatteryEnv
    
    results = {}
    
    # 1. Q-Learning
    print("Training Q-Learning Agent...")
    q_agent = SimpleQAgent()
    q_results = train_rl_agent(q_agent, BatteryEnv, data, n_episodes)
    
    eval_env = BatteryEnv(data.copy())
    q_eval_reward, q_history = q_agent.evaluate(eval_env)
    results['q_learning'] = {
        'training': q_results,
        'eval_reward': q_eval_reward,
        'history': q_history
    }
    
    # 2. DQN
    print("\nTraining DQN Agent...")
    dqn_agent = DQNAgent()
    dqn_results = train_rl_agent(dqn_agent, BatteryEnv, data, n_episodes)
    
    eval_env = BatteryEnv(data.copy())
    dqn_eval_reward, dqn_history = dqn_agent.evaluate(eval_env)
    results['dqn'] = {
        'training': dqn_results,
        'eval_reward': dqn_eval_reward,
        'history': dqn_history
    }
    
    # 3. MPC (理论上界)
    print("\nEvaluating MPC (Perfect Forecast)...")
    mpc_agent = MPCBaseline(horizon=24)
    mpc_agent.set_price_forecast(data['price'].tolist())
    
    eval_env = BatteryEnv(data.copy())
    obs, _ = eval_env.reset(options={"initial_soc": 0.5})
    mpc_reward = 0
    mpc_history = []
    
    while obs is not None:
        action = mpc_agent.decide(obs)
        next_obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        mpc_history.append({**obs, 'action': action, 'reward': reward})
        mpc_reward += reward
        obs = next_obs
        if done:
            break
    
    results['mpc'] = {
        'eval_reward': mpc_reward,
        'history': mpc_history
    }
    
    # 打印总结
    print("\n" + "="*50)
    print("Performance Comparison:")
    print("="*50)
    print(f"Q-Learning:  ${q_eval_reward:.2f}")
    print(f"DQN:         ${dqn_eval_reward:.2f}")
    print(f"MPC (Upper): ${mpc_reward:.2f}")
    print("="*50)
    
    return results
