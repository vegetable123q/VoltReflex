"""
处理 CAISO 真实数据
将 LMP 价格数据和负荷数据整合为项目可用的 1 小时数据
"""

import pandas as pd
import numpy as np
from pathlib import Path


def process_lmp_data(lmp_file: str) -> pd.DataFrame:
    """
    处理小时 LMP 价格数据
    
    Args:
        lmp_file: LMP 数据文件路径 (单节点小时数据)
    """
    print(f"读取 LMP 价格数据: {lmp_file}")
    df = pd.read_csv(lmp_file)
    
    # 显示节点信息
    unique_nodes = df['location'].unique()
    print(f"数据包含 {len(unique_nodes)} 个节点: {unique_nodes}")
    print(f"数据行数: {len(df)}")
    
    # 解析时间戳 (去除时区信息)
    df['timestamp'] = pd.to_datetime(df['interval_start_local']).dt.tz_localize(None)
    
    # 数据已经是小时级别，直接使用
    hourly_price = df[['timestamp', 'lmp', 'energy', 'congestion', 'loss']].copy()
    hourly_price = hourly_price.rename(columns={'lmp': 'price_mwh'})
    
    # 转换为 $/kWh（项目使用的单位）
    hourly_price['price'] = hourly_price['price_mwh'] / 1000
    
    print(f"LMP 数据处理完成: {len(hourly_price)} 个小时")
    print(f"价格范围: ${hourly_price['price'].min():.4f} - ${hourly_price['price'].max():.4f} /kWh")
    print(f"时间范围: {hourly_price['timestamp'].min()} 到 {hourly_price['timestamp'].max()}")
    
    return hourly_price


def process_load_data(load_file: str, tac_area: str = "CA ISO-TAC") -> pd.DataFrame:
    """
    处理小时负荷数据
    
    Args:
        load_file: 负荷数据文件路径
        tac_area: 传输区域名称，默认使用 "CA ISO-TAC"（CAISO 总负荷）
    """
    print(f"读取负荷数据: {load_file}")
    df = pd.read_csv(load_file)
    
    # 显示可用区域
    unique_areas = df['tac_area_name'].unique()
    print(f"数据包含 {len(unique_areas)} 个区域: {unique_areas[:10]}...")
    
    # 解析时间戳
    df['timestamp'] = pd.to_datetime(df['interval_start_local']).dt.tz_localize(None)
    
    # 筛选指定区域
    df_filtered = df[df['tac_area_name'] == tac_area].copy()
    
    if len(df_filtered) == 0:
        print(f"警告: 未找到区域 '{tac_area}'，使用所有区域总和")
        # 按时间戳汇总所有区域负荷
        df_filtered = df.groupby('timestamp').agg({'load': 'sum'}).reset_index()
    else:
        df_filtered = df_filtered[['timestamp', 'load']].copy()
    
    # 负荷单位转换: MW -> GW (便于显示)
    df_filtered['load_gw'] = df_filtered['load'] / 1000
    
    # 归一化负荷用于项目（缩放到合理范围 1-10）
    load_min = df_filtered['load'].min()
    load_max = df_filtered['load'].max()
    df_filtered['load_normalized'] = 1 + 9 * (df_filtered['load'] - load_min) / (load_max - load_min)
    
    print(f"负荷数据处理完成: {len(df_filtered)} 个小时")
    print(f"负荷范围: {df_filtered['load'].min():.0f} - {df_filtered['load'].max():.0f} MW")
    
    return df_filtered


def merge_data(price_df: pd.DataFrame, load_df: pd.DataFrame) -> pd.DataFrame:
    """
    合并价格和负荷数据
    """
    print("合并价格和负荷数据...")
    
    # 内连接确保数据完整
    merged = pd.merge(
        price_df[['timestamp', 'price', 'price_mwh']],
        load_df[['timestamp', 'load_normalized', 'load', 'load_gw']],
        on='timestamp',
        how='inner'
    )
    
    # 按时间排序
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    
    print(f"合并完成: {len(merged)} 个小时的数据")
    print(f"时间范围: {merged['timestamp'].min()} 到 {merged['timestamp'].max()}")
    
    return merged


def create_market_data(merged_df: pd.DataFrame, output_file: str):
    """
    创建项目可用的 market_data.csv 格式
    
    格式: timestamp, price, load
    - price: $/kWh
    - load: 归一化负荷 (1-10 范围)
    """
    output_df = merged_df[['timestamp', 'price', 'load_normalized']].copy()
    output_df = output_df.rename(columns={'load_normalized': 'load'})
    
    # 格式化时间戳
    output_df['timestamp'] = output_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 保存
    output_df.to_csv(output_file, index=False)
    print(f"\n✅ 已保存项目可用数据到: {output_file}")
    print(f"   - 数据行数: {len(output_df)}")
    print(f"   - 天数: {len(output_df) // 24}")
    
    # 显示数据样本
    print("\n数据样本:")
    print(output_df.head(10).to_string(index=False))
    
    # 显示统计信息
    print("\n统计信息:")
    print(f"  价格 ($/kWh):")
    print(f"    - 平均: ${output_df['price'].astype(float).mean():.4f}")
    print(f"    - 最小: ${output_df['price'].astype(float).min():.4f}")
    print(f"    - 最大: ${output_df['price'].astype(float).max():.4f}")
    print(f"  负荷 (归一化):")
    print(f"    - 平均: {output_df['load'].astype(float).mean():.2f}")
    print(f"    - 最小: {output_df['load'].astype(float).min():.2f}")
    print(f"    - 最大: {output_df['load'].astype(float).max():.2f}")
    
    return output_df


def save_full_data(merged_df: pd.DataFrame, output_file: str):
    """
    保存完整数据（包含原始值）用于分析
    """
    full_df = merged_df.copy()
    full_df['timestamp'] = full_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    full_df.to_csv(output_file, index=False)
    print(f"✅ 已保存完整数据到: {output_file}")


def main():
    # 文件路径
    data_dir = Path(__file__).parent
    lmp_file = data_dir / "CAISO_LMP_10TH_STW.csv"
    load_file = data_dir / "CAISO_load_hourly.csv"
    
    # 输出文件
    market_data_file = data_dir / "caiso_market_data.csv"
    full_data_file = data_dir / "caiso_full_data.csv"
    
    # 检查文件存在
    if not lmp_file.exists():
        print(f"❌ 未找到 LMP 数据文件: {lmp_file}")
        return
    if not load_file.exists():
        print(f"❌ 未找到负荷数据文件: {load_file}")
        return
    
    print("=" * 60)
    print("CAISO 数据处理")
    print("=" * 60)
    
    # 处理价格数据
    price_df = process_lmp_data(str(lmp_file))
    
    print()
    
    # 处理负荷数据 (使用 CA ISO-TAC 总负荷)
    load_df = process_load_data(str(load_file), tac_area="CA ISO-TAC")
    
    print()
    
    # 合并数据
    merged_df = merge_data(price_df, load_df)
    
    print()
    
    # 保存项目可用格式
    create_market_data(merged_df, str(market_data_file))
    
    # 保存完整数据
    save_full_data(merged_df, str(full_data_file))
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"\n使用方法: 在 configs/default.yaml 中设置:")
    print(f'  data:')
    print(f'    source: "custom"')
    print(f'    path: "data/caiso_market_data.csv"')


if __name__ == "__main__":
    main()
