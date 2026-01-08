"""
Battery Environment Module
类 Gym 环境，模拟家庭储能电池在分时电价市场中的行为
"""
from typing import Dict, Tuple, Literal
import pandas as pd


Action = Literal["CHARGE", "DISCHARGE", "HOLD"]


class BatteryEnv:
    """
    家庭储能电池环境
    
    模拟 Tesla Powerwall 参数:
    - 容量: 13.5 kWh
    - 最大功率: 5 kW
    - 往返效率: 90%
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化电池环境
        
        Args:
            df: 包含 timestamp, price, load 列的 DataFrame
        """
        self.df = df.reset_index(drop=True)
        
        # 电池参数 (Tesla Powerwall 规格)
        self.capacity = 13.5  # kWh
        self.max_power = 5.0  # kW
        self.efficiency = 0.9  # 往返效率 (round-trip)
        
        # 状态变量
        self.current_step = 0
        self.current_soc = 0.5  # 初始电量 50%
        
        # 记录历史
        self.history = []
    
    def reset(self, initial_soc: float = 0.5) -> Dict:
        """
        重置环境到初始状态
        
        Args:
            initial_soc: 初始电量百分比 (0-1)
            
        Returns:
            初始观测值
        """
        self.current_step = 0
        self.current_soc = initial_soc
        self.history = []
        return self.get_obs()
    
    def get_obs(self) -> Dict:
        """
        获取当前观测值
        
        Returns:
            包含 price, soc, hour, load, step 的字典
        """
        if self.current_step >= len(self.df):
            return None
            
        row = self.df.iloc[self.current_step]
        timestamp = pd.to_datetime(row['timestamp'])
        
        return {
            "price": float(row['price']),
            "soc": round(self.current_soc * 100, 1),  # 百分比
            "hour": timestamp.hour,
            "load": float(row['load']),
            "step": self.current_step,
            "day": self.current_step // 24,
            "timestamp": str(timestamp)
        }
    
    def step(self, action: Action) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一个动作并返回结果
        
        Args:
            action: "CHARGE", "DISCHARGE", or "HOLD"
            
        Returns:
            observation: 下一个状态观测
            reward: 本步骤的净利润 (负数表示花费)
            done: 是否结束
            info: 详细信息字典
        """
        obs = self.get_obs()
        if obs is None:
            return None, 0, True, {"error": "Episode finished"}
        
        price = obs['price']
        hour = obs['hour']
        load = obs['load']
        
        # 计算当前电量 (kWh)
        current_energy = self.current_soc * self.capacity
        
        # 初始化变量
        energy_charged = 0.0  # 充入电池的能量 (kWh)
        energy_discharged = 0.0  # 从电池放出的能量 (kWh)
        grid_cost = 0.0  # 从电网购电成本
        grid_revenue = 0.0  # 向电网售电收入
        
        action = action.upper().strip()
        
        if action == "CHARGE":
            # 充电: 从电网买电存入电池
            # 受 max_power 和剩余容量限制
            available_capacity = self.capacity - current_energy
            max_charge_energy = min(self.max_power, available_capacity)
            
            if max_charge_energy > 0:
                # 考虑充电效率 (充电损耗)
                # 从电网购买的能量 = 实际存入电池的能量 / 充电效率
                energy_charged = max_charge_energy
                energy_from_grid = energy_charged / (self.efficiency ** 0.5)  # 单向效率
                grid_cost = price * energy_from_grid
                
        elif action == "DISCHARGE":
            # 放电: 电池放电卖回电网或抵消负荷
            # 受 max_power 和当前电量限制
            min_soc_energy = 0.1 * self.capacity  # 保留 10% 最低电量
            available_energy = current_energy - min_soc_energy
            max_discharge_energy = min(self.max_power, max(0, available_energy))
            
            if max_discharge_energy > 0:
                # 考虑放电效率
                energy_discharged = max_discharge_energy
                energy_to_grid = energy_discharged * (self.efficiency ** 0.5)  # 单向效率
                grid_revenue = price * energy_to_grid
                
        # else: HOLD - 不做任何操作
        
        # 更新 SOC
        new_energy = current_energy + energy_charged - energy_discharged
        self.current_soc = max(0.0, min(1.0, new_energy / self.capacity))
        
        # 计算奖励 (净利润)
        reward = grid_revenue - grid_cost
        
        # 记录历史
        info = {
            "action": action,
            "price": price,
            "hour": hour,
            "load": load,
            "soc_before": round((current_energy / self.capacity) * 100, 1),
            "soc_after": round(self.current_soc * 100, 1),
            "energy_charged": round(energy_charged, 3),
            "energy_discharged": round(energy_discharged, 3),
            "grid_cost": round(grid_cost, 4),
            "grid_revenue": round(grid_revenue, 4),
            "reward": round(reward, 4),
            "step": self.current_step,
            "day": self.current_step // 24
        }
        self.history.append(info)
        
        # 移动到下一步
        self.current_step += 1
        done = self.current_step >= len(self.df)
        
        next_obs = self.get_obs() if not done else None
        
        return next_obs, reward, done, info
    
    def get_daily_summary(self, day: int) -> Dict:
        """
        获取某一天的交易摘要
        
        Args:
            day: 天数索引 (从0开始)
            
        Returns:
            包含当天统计信息的字典
        """
        day_records = [h for h in self.history if h['day'] == day]
        
        if not day_records:
            return {"day": day, "total_profit": 0, "transactions": 0}
        
        total_profit = sum(h['reward'] for h in day_records)
        total_cost = sum(h['grid_cost'] for h in day_records)
        total_revenue = sum(h['grid_revenue'] for h in day_records)
        charge_count = sum(1 for h in day_records if h['action'] == 'CHARGE')
        discharge_count = sum(1 for h in day_records if h['action'] == 'DISCHARGE')
        
        return {
            "day": day,
            "total_profit": round(total_profit, 4),
            "total_cost": round(total_cost, 4),
            "total_revenue": round(total_revenue, 4),
            "charge_count": charge_count,
            "discharge_count": discharge_count,
            "hold_count": len(day_records) - charge_count - discharge_count,
            "records": day_records
        }
    
    def get_total_profit(self) -> float:
        """获取总利润"""
        return sum(h['reward'] for h in self.history)
    
    @property
    def total_steps(self) -> int:
        """总步数"""
        return len(self.df)
    
    @property
    def total_days(self) -> int:
        """总天数"""
        return len(self.df) // 24


if __name__ == "__main__":
    # 简单测试
    import os
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'caiso_enhanced_data.csv')
    df = pd.read_csv(data_path)
    
    env = BatteryEnv(df)
    obs = env.reset()
    
    print("初始状态:", obs)
    print(f"总天数: {env.total_days}, 总步数: {env.total_steps}")
    
    # 测试几个动作
    for action in ["CHARGE", "DISCHARGE", "HOLD"]:
        obs = env.reset()
        next_obs, reward, done, info = env.step(action)
        print(f"\n动作 {action}:")
        print(f"  奖励: ${reward:.4f}")
        print(f"  SOC: {info['soc_before']}% -> {info['soc_after']}%")
