"""
生成合成市场数据的脚本
模拟14天（336小时）的电价和负荷数据
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_market_data(days=14, output_path="market_data.csv"):
    """
    生成合成的电力市场数据
    
    Args:
        days: 模拟天数（默认14天）
        output_path: 输出文件路径
    """
    hours = days * 24
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    
    timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
    prices = []
    loads = []
    
    for ts in timestamps:
        hour = ts.hour
        
        # 定义电价规则
        # Peak hours (17:00-21:00): 高电价 0.45-0.55 $/kWh
        # Off-peak hours (23:00-07:00): 低电价 0.08-0.12 $/kWh
        # Shoulder hours: 中等电价 0.18-0.22 $/kWh
        if 17 <= hour <= 20:  # Peak
            base_price = 0.50
            noise = np.random.uniform(-0.05, 0.05)
            price = base_price + noise
        elif 23 <= hour or hour <= 6:  # Off-peak
            base_price = 0.10
            noise = np.random.uniform(-0.02, 0.02)
            price = base_price + noise
        else:  # Shoulder
            base_price = 0.20
            noise = np.random.uniform(-0.02, 0.02)
            price = base_price + noise
        
        # 定义负荷曲线（家庭用电模式）
        # 早高峰 (07:00-09:00): 3-4 kW
        # 晚高峰 (18:00-22:00): 4-6 kW
        # 夜间 (23:00-06:00): 1-2 kW
        # 其他时间: 2-3 kW
        if 7 <= hour <= 8:  # Morning peak
            base_load = 3.5
            noise = np.random.uniform(-0.5, 0.5)
            load = base_load + noise
        elif 18 <= hour <= 21:  # Evening peak
            base_load = 5.0
            noise = np.random.uniform(-1.0, 1.0)
            load = base_load + noise
        elif 23 <= hour or hour <= 5:  # Night
            base_load = 1.5
            noise = np.random.uniform(-0.3, 0.3)
            load = base_load + noise
        else:  # Other hours
            base_load = 2.5
            noise = np.random.uniform(-0.5, 0.5)
            load = base_load + noise
        
        # 确保非负
        price = max(0.05, price)
        load = max(0.5, load)
        
        prices.append(round(price, 4))
        loads.append(round(load, 2))
    
    # 创建 DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'load': loads
    })
    
    # 保存到 CSV
    df.to_csv(output_path, index=False)
    print(f"✅ 成功生成 {hours} 小时的市场数据")
    print(f"📊 价格范围: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"⚡ 负荷范围: {df['load'].min():.2f} kW - {df['load'].max():.2f} kW")
    print(f"💾 数据已保存到: {output_path}")
    
    return df

if __name__ == "__main__":
    import os
    # 确保输出到 data 目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "market_data.csv")
    
    df = generate_market_data(days=14, output_path=output_file)
    
    # 打印统计摘要
    print("\n" + "="*50)
    print("数据统计摘要:")
    print("="*50)
    print(df.describe())
