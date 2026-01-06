"""
真实电价数据加载器
支持 CAISO, PJM, ERCOT 等市场数据
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import requests
from io import StringIO


class RealDataLoader:
    """
    真实电价数据加载器
    
    支持的数据源:
    - CAISO (California ISO)
    - PJM (Pennsylvania-New Jersey-Maryland)
    - ERCOT (Texas)
    - 本地 CSV 文件
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_caiso_data(
        self,
        start_date: str,
        end_date: str,
        node: str = "TH_SP15_GEN-APND"
    ) -> pd.DataFrame:
        """
        加载 CAISO 日前市场价格数据
        
        注意: 实际使用需要 CAISO OASIS API 账号
        这里提供模拟数据结构
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期
            node: 定价节点 (默认为 SP15 区域)
        """
        # 检查缓存
        cache_file = os.path.join(
            self.cache_dir, 
            f"caiso_{start_date}_{end_date}_{node}.csv"
        )
        
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, parse_dates=['timestamp'])
        
        # 生成 CAISO 风格的模拟数据
        # 实际项目中应替换为真实 API 调用
        df = self._generate_caiso_style_data(start_date, end_date)
        
        # 缓存
        df.to_csv(cache_file, index=False)
        
        return df
    
    def _generate_caiso_style_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        生成符合 CAISO 特征的模拟数据
        
        CAISO 特点:
        - 高太阳能渗透导致中午价格低（鸭子曲线）
        - 傍晚太阳能退出时价格急剧上升
        - 夏季下午高峰更明显
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        hours = int((end - start).total_seconds() / 3600)
        
        timestamps = [start + timedelta(hours=i) for i in range(hours)]
        prices = []
        loads = []
        
        for ts in timestamps:
            hour = ts.hour
            month = ts.month
            day_of_week = ts.weekday()
            
            # 基础价格模式（鸭子曲线）
            if 10 <= hour <= 15:  # 太阳能高峰 -> 低价
                base_price = 0.05 + 0.03 * np.random.random()
                # 有时甚至出现负价格
                if np.random.random() < 0.1:
                    base_price = -0.02
            elif 17 <= hour <= 21:  # 傍晚高峰 -> 高价
                base_price = 0.35 + 0.25 * np.random.random()
                # 极端高峰
                if np.random.random() < 0.05:
                    base_price = 0.80 + 0.40 * np.random.random()
            elif 6 <= hour <= 9:  # 早高峰
                base_price = 0.15 + 0.10 * np.random.random()
            elif 22 <= hour or hour <= 5:  # 夜间
                base_price = 0.08 + 0.04 * np.random.random()
            else:  # 其他时段
                base_price = 0.12 + 0.06 * np.random.random()
            
            # 季节性调整（夏季更高）
            if 6 <= month <= 9:
                base_price *= 1.3
            
            # 周末略低
            if day_of_week >= 5:
                base_price *= 0.85
            
            # 添加随机波动
            price = max(-0.05, base_price + np.random.normal(0, 0.02))
            
            # 负荷模式
            if 6 <= hour <= 9:
                load = 4.5 + np.random.random()
            elif 17 <= hour <= 21:
                load = 5.5 + 1.5 * np.random.random()
            elif 22 <= hour or hour <= 5:
                load = 1.5 + 0.5 * np.random.random()
            else:
                load = 3.0 + np.random.random()
            
            prices.append(round(price, 4))
            loads.append(round(load, 2))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'load': loads,
            'source': 'CAISO_simulated'
        })
    
    def load_pjm_data(
        self,
        start_date: str,
        end_date: str,
        zone: str = "PECO"
    ) -> pd.DataFrame:
        """
        加载 PJM 市场数据
        
        PJM 特点:
        - 东海岸市场，冬季供暖负荷高
        - 价格波动相对稳定
        """
        cache_file = os.path.join(
            self.cache_dir,
            f"pjm_{start_date}_{end_date}_{zone}.csv"
        )
        
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, parse_dates=['timestamp'])
        
        df = self._generate_pjm_style_data(start_date, end_date)
        df.to_csv(cache_file, index=False)
        
        return df
    
    def _generate_pjm_style_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """生成 PJM 风格数据"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        hours = int((end - start).total_seconds() / 3600)
        
        timestamps = [start + timedelta(hours=i) for i in range(hours)]
        prices = []
        loads = []
        
        for ts in timestamps:
            hour = ts.hour
            month = ts.month
            
            # PJM 价格模式（更传统的双峰）
            if 7 <= hour <= 10:  # 早高峰
                base_price = 0.08 + 0.04 * np.random.random()
            elif 16 <= hour <= 20:  # 晚高峰
                base_price = 0.12 + 0.06 * np.random.random()
            elif 23 <= hour or hour <= 5:  # 低谷
                base_price = 0.03 + 0.02 * np.random.random()
            else:
                base_price = 0.05 + 0.03 * np.random.random()
            
            # 冬季供暖高峰
            if month in [12, 1, 2] and 6 <= hour <= 9:
                base_price *= 1.4
            
            # 夏季空调高峰
            if month in [6, 7, 8] and 14 <= hour <= 18:
                base_price *= 1.3
            
            price = max(0.01, base_price + np.random.normal(0, 0.01))
            
            # 负荷
            if 6 <= hour <= 9:
                load = 4.0 + np.random.random()
            elif 17 <= hour <= 21:
                load = 5.0 + np.random.random()
            else:
                load = 2.5 + np.random.random()
            
            prices.append(round(price, 4))
            loads.append(round(load, 2))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'load': loads,
            'source': 'PJM_simulated'
        })
    
    def load_ercot_data(
        self,
        start_date: str,
        end_date: str,
        zone: str = "HB_HOUSTON"
    ) -> pd.DataFrame:
        """
        加载 ERCOT (德州) 市场数据
        
        ERCOT 特点:
        - 能源孤岛，价格波动极大
        - 夏季空调负荷导致极端高价
        - 偶发负价格（风电过剩）
        """
        cache_file = os.path.join(
            self.cache_dir,
            f"ercot_{start_date}_{end_date}_{zone}.csv"
        )
        
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, parse_dates=['timestamp'])
        
        df = self._generate_ercot_style_data(start_date, end_date)
        df.to_csv(cache_file, index=False)
        
        return df
    
    def _generate_ercot_style_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """生成 ERCOT 风格数据（高波动性）"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        hours = int((end - start).total_seconds() / 3600)
        
        timestamps = [start + timedelta(hours=i) for i in range(hours)]
        prices = []
        loads = []
        
        for ts in timestamps:
            hour = ts.hour
            month = ts.month
            
            # ERCOT 基础模式
            if 14 <= hour <= 19:  # 下午高峰
                base_price = 0.08 + 0.04 * np.random.random()
            elif 0 <= hour <= 5:  # 夜间低谷
                base_price = 0.02 + 0.02 * np.random.random()
                # 风电过剩可能导致负价格
                if np.random.random() < 0.15:
                    base_price = -0.01 - 0.02 * np.random.random()
            else:
                base_price = 0.04 + 0.03 * np.random.random()
            
            # 夏季极端高价
            if month in [6, 7, 8] and 14 <= hour <= 18:
                if np.random.random() < 0.1:
                    # 极端价格尖峰
                    base_price = 1.0 + 2.0 * np.random.random()
                else:
                    base_price *= 2.0
            
            price = base_price + np.random.normal(0, 0.02)
            
            # 负荷
            if 14 <= hour <= 19:
                load = 6.0 + 2.0 * np.random.random()
            else:
                load = 2.5 + 1.0 * np.random.random()
            
            prices.append(round(price, 4))
            loads.append(round(load, 2))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'load': loads,
            'source': 'ERCOT_simulated'
        })
    
    def load_from_csv(
        self,
        file_path: str,
        timestamp_col: str = 'timestamp',
        price_col: str = 'price',
        load_col: Optional[str] = 'load'
    ) -> pd.DataFrame:
        """
        从 CSV 文件加载自定义数据
        """
        df = pd.read_csv(file_path)
        
        # 标准化列名
        df = df.rename(columns={
            timestamp_col: 'timestamp',
            price_col: 'price',
        })
        
        if load_col and load_col in df.columns:
            df = df.rename(columns={load_col: 'load'})
        else:
            df['load'] = 3.0  # 默认负荷
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['source'] = 'custom'
        
        return df[['timestamp', 'price', 'load', 'source']]
    
    def add_price_spikes(
        self,
        df: pd.DataFrame,
        spike_probability: float = 0.02,
        spike_multiplier: float = 3.0
    ) -> pd.DataFrame:
        """
        向数据中添加价格尖峰事件
        用于测试 Agent 对异常情况的响应
        """
        df = df.copy()
        
        n = len(df)
        spike_mask = np.random.random(n) < spike_probability
        
        df.loc[spike_mask, 'price'] *= spike_multiplier
        df.loc[spike_mask, 'is_spike'] = True
        df['is_spike'] = df.get('is_spike', False).fillna(False)
        
        return df
    
    def add_negative_prices(
        self,
        df: pd.DataFrame,
        negative_probability: float = 0.05,
        negative_range: Tuple[float, float] = (-0.05, 0)
    ) -> pd.DataFrame:
        """
        添加负电价事件（模拟可再生能源过剩）
        """
        df = df.copy()
        
        n = len(df)
        negative_mask = np.random.random(n) < negative_probability
        
        # 只在低负荷时段添加负价格
        hour_mask = df['timestamp'].dt.hour.isin([0, 1, 2, 3, 4, 5, 12, 13, 14])
        combined_mask = negative_mask & hour_mask
        
        negative_prices = np.random.uniform(
            negative_range[0], 
            negative_range[1], 
            size=combined_mask.sum()
        )
        
        df.loc[combined_mask, 'price'] = negative_prices
        df.loc[combined_mask, 'is_negative'] = True
        df['is_negative'] = df.get('is_negative', False).fillna(False)
        
        return df


def load_data_by_config(config: dict) -> pd.DataFrame:
    """
    根据配置文件加载数据
    """
    loader = RealDataLoader()
    source = config.get('source', 'synthetic')
    
    if source == 'synthetic':
        # 使用现有的合成数据生成器
        from data.generate_data import generate_market_data
        days = config.get('synthetic', {}).get('days', 14)
        return generate_market_data(days=days, output_path=None)
    
    elif source == 'caiso':
        real_config = config.get('real_data', {})
        return loader.load_caiso_data(
            start_date=real_config.get('start_date', '2024-01-01'),
            end_date=real_config.get('end_date', '2024-01-14')
        )
    
    elif source == 'pjm':
        real_config = config.get('real_data', {})
        return loader.load_pjm_data(
            start_date=real_config.get('start_date', '2024-01-01'),
            end_date=real_config.get('end_date', '2024-01-14')
        )
    
    elif source == 'ercot':
        real_config = config.get('real_data', {})
        return loader.load_ercot_data(
            start_date=real_config.get('start_date', '2024-01-01'),
            end_date=real_config.get('end_date', '2024-01-14')
        )
    
    elif source == 'custom':
        return loader.load_from_csv(config.get('path'))
    
    else:
        raise ValueError(f"Unknown data source: {source}")
