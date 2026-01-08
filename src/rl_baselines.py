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
    Model Predictive Control 基线 - 理论最优上界
    
    使用线性规划求解完美预见下的最优套利策略。
    假设未来价格完全已知，计算全局最优的充放电计划。
    
    这是一个理论上界，实际策略不可能达到（因为无法预知未来价格）。
    """
    
    def __init__(
        self, 
        horizon: int = 24,
        capacity_kwh: float = 13.5,
        max_charge_kw: float = 5.0,
        max_discharge_kw: float = 5.0,
        efficiency: float = 0.9,
        min_soc: float = 0.10,
        max_soc: float = 0.95,
    ):
        self.horizon = horizon
        self.capacity_kwh = capacity_kwh
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.efficiency = efficiency  # round-trip efficiency
        self.eff_charge = np.sqrt(efficiency)
        self.eff_discharge = np.sqrt(efficiency)
        self.min_soc = min_soc
        self.max_soc = max_soc
        
        self.future_prices = None
        self.current_index = 0
        self.optimal_actions = None  # 预计算的最优动作序列
    
    def set_price_forecast(self, prices: List[float], initial_soc: float = 0.5):
        """
        设置价格预测并求解全局最优策略
        
        Args:
            prices: 完整的价格序列
            initial_soc: 初始 SOC (0-1)
        """
        self.future_prices = prices
        self.current_index = 0
        
        # 使用线性规划求解最优动作序列
        self.optimal_actions = self._solve_optimal_schedule(prices, initial_soc)
    
    def _solve_optimal_schedule(
        self, 
        prices: List[float], 
        initial_soc: float
    ) -> List[str]:
        """
        使用线性规划求解最优充放电计划
        
        决策变量:
        - charge[t]: 第t时刻充电功率 (kW), >= 0
        - discharge[t]: 第t时刻放电功率 (kW), >= 0
        - soc[t]: 第t时刻的 SOC
        
        目标: 最大化 sum_t (discharge[t] * price[t] * eff_discharge - charge[t] * price[t] / eff_charge)
        
        约束:
        - soc[t+1] = soc[t] + charge[t] * eff_charge / capacity - discharge[t] / (capacity * eff_discharge)
        - min_soc <= soc[t] <= max_soc
        - 0 <= charge[t] <= max_charge
        - 0 <= discharge[t] <= max_discharge
        """
        try:
            from scipy.optimize import linprog
        except ImportError:
            print("Warning: scipy not available, falling back to heuristic MPC")
            return self._heuristic_schedule(prices, initial_soc)
        
        T = len(prices)
        if T == 0:
            return []
        
        # 决策变量: [charge_0, ..., charge_{T-1}, discharge_0, ..., discharge_{T-1}]
        # 共 2T 个变量
        
        # 目标函数系数 (最大化利润 => 最小化负利润)
        # profit = sum_t (discharge[t] * price[t] * eff_discharge - charge[t] * price[t] / eff_charge)
        c = []
        for t in range(T):
            # charge[t] 的系数: +price[t] / eff_charge (成本，要最小化)
            c.append(prices[t] / self.eff_charge)
        for t in range(T):
            # discharge[t] 的系数: -price[t] * eff_discharge (收益的负值)
            c.append(-prices[t] * self.eff_discharge)
        
        c = np.array(c)
        
        # SOC 动态约束: soc[t+1] = soc[t] + charge[t] * eff / cap - discharge[t] / (cap * eff)
        # 重写为: soc[t+1] - soc[t] - charge[t] * eff / cap + discharge[t] / (cap * eff) = 0
        # 
        # 我们用 soc[t] = soc[0] + sum_{i=0}^{t-1} (charge[i] * eff / cap - discharge[i] / (cap * eff))
        # 所以 soc[t] 是 charge 和 discharge 的线性函数
        
        # SOC 边界约束 (作为不等式约束)
        # min_soc <= soc[t] <= max_soc for all t
        # 
        # soc[t] = soc_init + sum_{i<t} delta_i
        # where delta_i = charge[i] * eff_charge / cap - discharge[i] / (cap * eff_discharge)
        
        cap = self.capacity_kwh
        eff_c = self.eff_charge
        eff_d = self.eff_discharge
        
        A_ub = []
        b_ub = []
        
        # 对于每个时刻 t = 1, ..., T, 构建 SOC 约束
        # soc[t] = soc_init + sum_{i=0}^{t-1} (charge[i] * eff_c / cap - discharge[i] / (cap * eff_d))
        # 
        # 上界约束: soc[t] <= max_soc
        # => sum_{i<t} (charge[i] * eff_c / cap) - sum_{i<t} (discharge[i] / (cap * eff_d)) <= max_soc - soc_init
        #
        # 下界约束: soc[t] >= min_soc
        # => -sum_{i<t} (charge[i] * eff_c / cap) + sum_{i<t} (discharge[i] / (cap * eff_d)) <= soc_init - min_soc
        
        for t in range(1, T + 1):
            # 上界约束: sum_{i<t} charge[i] * (eff_c/cap) - sum_{i<t} discharge[i] * (1/(cap*eff_d)) <= max_soc - soc_init
            row_upper = np.zeros(2 * T)
            for i in range(t):
                row_upper[i] = eff_c / cap  # charge[i]
                row_upper[T + i] = -1.0 / (cap * eff_d)  # discharge[i]
            A_ub.append(row_upper)
            b_ub.append(self.max_soc - initial_soc)
            
            # 下界约束: -sum_{i<t} charge[i] * (eff_c/cap) + sum_{i<t} discharge[i] * (1/(cap*eff_d)) <= soc_init - min_soc
            row_lower = np.zeros(2 * T)
            for i in range(t):
                row_lower[i] = -eff_c / cap  # charge[i]
                row_lower[T + i] = 1.0 / (cap * eff_d)  # discharge[i]
            A_ub.append(row_lower)
            b_ub.append(initial_soc - self.min_soc)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # 变量边界
        bounds = []
        for _ in range(T):
            bounds.append((0, self.max_charge_kw))  # charge bounds
        for _ in range(T):
            bounds.append((0, self.max_discharge_kw))  # discharge bounds
        
        # 求解线性规划
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if result.success:
                charge = result.x[:T]
                discharge = result.x[T:]
                
                # 转换为离散动作
                actions = []
                for t in range(T):
                    if charge[t] > 0.5:  # 充电阈值
                        actions.append('CHARGE')
                    elif discharge[t] > 0.5:  # 放电阈值
                        actions.append('DISCHARGE')
                    else:
                        actions.append('HOLD')
                
                return actions
            else:
                print(f"LP optimization failed: {result.message}, using heuristic")
                return self._heuristic_schedule(prices, initial_soc)
                
        except Exception as e:
            print(f"LP solver error: {e}, using heuristic")
            return self._heuristic_schedule(prices, initial_soc)
    
    def _heuristic_schedule(
        self, 
        prices: List[float], 
        initial_soc: float
    ) -> List[str]:
        """
        备用启发式方法：识别价格谷值和峰值进行套利
        """
        T = len(prices)
        if T == 0:
            return []
        
        actions = []
        soc = initial_soc
        
        # 计算每日的价格统计
        window_size = min(24, T)
        
        for t in range(T):
            price = prices[t]
            
            # 计算当前窗口的价格范围
            window_start = max(0, t - window_size // 2)
            window_end = min(T, t + window_size // 2)
            window_prices = prices[window_start:window_end]
            
            if len(window_prices) > 0:
                p_min = min(window_prices)
                p_max = max(window_prices)
                p_range = p_max - p_min
                
                # 归一化当前价格位置
                if p_range > 1e-6:
                    price_percentile = (price - p_min) / p_range
                else:
                    price_percentile = 0.5
                
                # 基于价格百分位和SOC状态决策
                if price_percentile < 0.25 and soc < self.max_soc - 0.05:
                    # 价格在低25%，充电
                    actions.append('CHARGE')
                    soc = min(self.max_soc, soc + self.max_charge_kw * self.eff_charge / self.capacity_kwh)
                elif price_percentile > 0.75 and soc > self.min_soc + 0.05:
                    # 价格在高25%，放电
                    actions.append('DISCHARGE')
                    soc = max(self.min_soc, soc - self.max_discharge_kw / (self.capacity_kwh * self.eff_discharge))
                else:
                    actions.append('HOLD')
            else:
                actions.append('HOLD')
        
        return actions
    
    def decide(self, obs: Dict) -> str:
        """基于预计算的最优计划返回动作"""
        if self.optimal_actions is None or self.current_index >= len(self.optimal_actions):
            # 如果没有预计算，使用简单启发式
            return self._simple_decide(obs)
        
        action = self.optimal_actions[self.current_index]
        self.current_index += 1
        
        # 安全检查 - 确保不违反 SOC 约束
        soc = obs.get('soc', 50) / 100.0 if obs.get('soc', 50) > 1 else obs.get('soc', 0.5)
        if action == 'CHARGE' and soc >= self.max_soc - 0.01:
            return 'HOLD'
        if action == 'DISCHARGE' and soc <= self.min_soc + 0.01:
            return 'HOLD'
        
        return action
    
    def _simple_decide(self, obs: Dict) -> str:
        """简单的启发式决策（备用）"""
        if self.future_prices is None:
            return 'HOLD'
        
        current_price = obs['price']
        soc = obs.get('soc', 50)
        if soc > 1:  # 如果是百分比形式
            soc = soc / 100.0
        
        # 获取未来价格窗口
        future_end = min(self.current_index + self.horizon, len(self.future_prices))
        future_window = self.future_prices[self.current_index:future_end]
        
        if len(future_window) == 0:
            self.current_index += 1
            return 'HOLD'
        
        max_future = max(future_window)
        min_future = min(future_window)
        
        # 如果当前价格是未来窗口最低的10%，充电
        # 如果当前价格是未来窗口最高的10%，放电
        if current_price <= min_future * 1.05 and soc < self.max_soc - 0.05:
            action = 'CHARGE'
        elif current_price >= max_future * 0.95 and soc > self.min_soc + 0.05:
            action = 'DISCHARGE'
        else:
            action = 'HOLD'
        
        self.current_index += 1
        return action
    
    def reset(self):
        """重置"""
        self.current_index = 0
        self.optimal_actions = None


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
