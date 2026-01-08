"""
增强 CAISO 价格数据波动
保留原始数据的时间模式特征，同时增大价格波动幅度以增加套利空间
"""

import pandas as pd
import numpy as np
from pathlib import Path


def enhance_price_volatility(
    df: pd.DataFrame,
    volatility_scale: float = 3.0,
    peak_multiplier: float = 2.5,
    valley_multiplier: float = 0.5,
    spike_probability: float = 0.03,
    spike_multiplier: float = 4.0,
    negative_price_prob: float = 0.02,
    seed: int = 42
) -> pd.DataFrame:
    """
    增强价格波动，保留原始特征
    
    Args:
        df: 原始数据 DataFrame
        volatility_scale: 波动放大倍数 (围绕均值拉伸)
        peak_multiplier: 高峰时段价格乘数
        valley_multiplier: 低谷时段价格乘数
        spike_probability: 价格尖峰概率
        spike_multiplier: 尖峰价格乘数
        negative_price_prob: 负电价概率 (模拟可再生能源过剩)
        seed: 随机种子
    
    Returns:
        增强后的 DataFrame
    """
    np.random.seed(seed)
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # 计算原始均值
    original_mean = df['price'].mean()
    
    # ========================================
    # 1. 基于时段的价格调整
    # ========================================
    # 定义时段: 低谷(0-5, 12-14), 高峰(6-9, 16-21), 平时(其他)
    def get_time_multiplier(hour):
        if hour in [0, 1, 2, 3, 4, 5, 12, 13, 14]:  # 低谷
            return valley_multiplier
        elif hour in [6, 7, 8, 9, 16, 17, 18, 19, 20, 21]:  # 高峰
            return peak_multiplier
        else:  # 平时
            return 1.0
    
    df['time_mult'] = df['hour'].apply(get_time_multiplier)
    
    # ========================================
    # 2. 围绕均值拉伸波动
    # ========================================
    # 计算偏离均值的程度，并放大
    df['deviation'] = df['price'] - original_mean
    df['enhanced_price'] = original_mean + df['deviation'] * volatility_scale
    
    # 应用时段乘数
    df['enhanced_price'] = df['enhanced_price'] * df['time_mult']
    
    # ========================================
    # 3. 添加日内随机波动
    # ========================================
    noise = np.random.normal(0, original_mean * 0.15, len(df))
    df['enhanced_price'] += noise
    
    # ========================================
    # 4. 添加价格尖峰事件
    # ========================================
    spike_mask = np.random.random(len(df)) < spike_probability
    # 尖峰主要发生在高峰时段
    peak_hours = df['hour'].isin([17, 18, 19, 20])
    spike_mask = spike_mask & peak_hours
    df.loc[spike_mask, 'enhanced_price'] *= spike_multiplier
    df['is_spike'] = spike_mask
    
    # ========================================
    # 5. 添加负电价 (可再生能源过剩)
    # ========================================
    negative_mask = np.random.random(len(df)) < negative_price_prob
    # 负电价主要发生在低谷时段
    valley_hours = df['hour'].isin([0, 1, 2, 3, 12, 13, 14])
    negative_mask = negative_mask & valley_hours
    df.loc[negative_mask, 'enhanced_price'] = np.random.uniform(-0.02, -0.005, negative_mask.sum())
    df['is_negative'] = negative_mask
    
    # ========================================
    # 6. 确保价格合理性
    # ========================================
    # 设置价格下限 (除了允许的负电价)
    df.loc[~df['is_negative'], 'enhanced_price'] = df.loc[~df['is_negative'], 'enhanced_price'].clip(lower=0.01)
    # 设置价格上限
    df['enhanced_price'] = df['enhanced_price'].clip(upper=1.0)
    
    # 重命名并清理
    df['price'] = df['enhanced_price']
    df = df[['timestamp', 'price', 'load', 'is_spike', 'is_negative']]
    
    return df


def main():
    data_dir = Path(__file__).parent
    input_file = data_dir / "caiso_market_data.csv"
    output_file = data_dir / "caiso_enhanced_data.csv"
    
    print("=" * 60)
    print("CAISO 价格数据增强")
    print("=" * 60)
    
    # 加载原始数据
    print(f"\n读取原始数据: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"\n=== 原始数据统计 ===")
    print(f"  数据量: {len(df)} 小时")
    print(f"  价格均值: ${df['price'].mean():.4f}")
    print(f"  价格标准差: ${df['price'].std():.4f}")
    print(f"  价格范围: ${df['price'].min():.4f} - ${df['price'].max():.4f}")
    print(f"  峰谷差: ${df['price'].max() - df['price'].min():.4f}")
    
    # 增强数据
    print("\n正在增强价格波动...")
    enhanced_df = enhance_price_volatility(
        df,
        volatility_scale=3.0,       # 波动放大3倍
        peak_multiplier=2.5,        # 高峰时段价格×2.5
        valley_multiplier=0.5,      # 低谷时段价格×0.5
        spike_probability=0.03,     # 3%概率出现尖峰
        spike_multiplier=4.0,       # 尖峰价格×4
        negative_price_prob=0.02,   # 2%概率负电价
    )
    
    print(f"\n=== 增强后数据统计 ===")
    print(f"  数据量: {len(enhanced_df)} 小时")
    print(f"  价格均值: ${enhanced_df['price'].mean():.4f}")
    print(f"  价格标准差: ${enhanced_df['price'].std():.4f}")
    print(f"  价格范围: ${enhanced_df['price'].min():.4f} - ${enhanced_df['price'].max():.4f}")
    print(f"  峰谷差: ${enhanced_df['price'].max() - enhanced_df['price'].min():.4f}")
    print(f"  尖峰事件: {enhanced_df['is_spike'].sum()} 次")
    print(f"  负电价事件: {enhanced_df['is_negative'].sum()} 次")
    
    # 显示每小时平均价格对比
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    enhanced_df['hour'] = pd.to_datetime(enhanced_df['timestamp']).dt.hour
    
    print("\n=== 每小时平均价格对比 ===")
    print(f"{'Hour':<6} {'原始':>10} {'增强':>10} {'变化':>10}")
    print("-" * 40)
    for h in range(24):
        orig = df[df['hour'] == h]['price'].mean()
        enh = enhanced_df[enhanced_df['hour'] == h]['price'].mean()
        change = (enh - orig) / orig * 100 if orig > 0 else 0
        bar = '█' * int(enh * 20)
        print(f"{h:02d}:00  ${orig:.4f}   ${enh:.4f}   {change:+.1f}%  {bar}")
    
    # 保存输出文件 (只保留核心列)
    output_df = enhanced_df[['timestamp', 'price', 'load']].copy()
    output_df['timestamp'] = pd.to_datetime(output_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    output_df.to_csv(output_file, index=False)
    
    print(f"\n✅ 增强数据已保存: {output_file}")
    
    # 显示样本数据
    print("\n=== 样本数据 ===")
    print(output_df.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("使用方法: 在 configs/default.yaml 中设置:")
    print('  data:')
    print('    source: "custom"')
    print('    path: "data/caiso_enhanced_data.csv"')
    print("=" * 60)


if __name__ == "__main__":
    main()
