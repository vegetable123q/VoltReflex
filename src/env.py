"""
Battery arbitrage environment (Gymnasium).

MDP (incremental arbitrage reward):
- Exogenous series: (timestamp, base_price, load_kwh[, buy/sell prices])
- State s_t: {price, buy_price, sell_price, load_kwh, soc, hour, day, step}
- Action a_t (default discrete): {HOLD, CHARGE, DISCHARGE}
- Battery dynamics: capacity/power/efficiency, SOC bounds.
- Settlement:
    baseline_cost = load_kwh * buy_price
    with battery: charging imports energy_from_grid; discharging offsets load first, then (optionally) exports.
    reward = baseline_cost - (grid_import*buy_price - export*sell_price)
            = avoided_cost + export_revenue - charge_cost
  => HOLD gives 0 reward, so returns measure incremental arbitrage benefit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise ImportError(
        "gymnasium is required for src.env.BatteryEnv. "
        "Install it via `pip install gymnasium`."
    ) from e


ActionStr = Literal["CHARGE", "DISCHARGE", "HOLD"]
Action = Union[int, float, str]


@dataclass(frozen=True)
class BatteryEnvConfig:
    # Battery physics
    capacity_kwh: float = 13.5
    max_charge_kw: float = 5.0
    max_discharge_kw: float = 5.0
    dt_hours: float = 1.0
    roundtrip_efficiency: float = 0.9
    min_soc: float = 0.10
    max_soc: float = 0.95

    # Market / settlement
    allow_grid_charge: bool = True
    allow_export: bool = True
    buy_price_multiplier: float = 1.0
    buy_price_spread: float = 0.0
    sell_price_multiplier: float = 1.0
    sell_price_spread: float = 0.0
    # If data has explicit columns, they override the multipliers/spreads.
    buy_price_col: str = "price_buy"
    sell_price_col: str = "price_sell"
    base_price_col: str = "price"

    # Load handling
    load_kwh_col: str = "load_kwh"
    load_col: str = "load"
    load_profile_daily_kwh: float = 30.0  # used if no usable load column
    use_load_column_as_kwh: bool = False

    # RL API
    action_mode: Literal["discrete", "continuous"] = "discrete"

    @property
    def eff_charge(self) -> float:
        return float(np.sqrt(self.roundtrip_efficiency))

    @property
    def eff_discharge(self) -> float:
        return float(np.sqrt(self.roundtrip_efficiency))


def _action_to_power_kw(action: Action, cfg: BatteryEnvConfig) -> float:
    # Discrete mapping (for legacy agents)
    if isinstance(action, str):
        a = action.upper().strip()
        if a == "CHARGE":
            return cfg.max_charge_kw
        if a == "DISCHARGE":
            return -cfg.max_discharge_kw
        return 0.0

    if isinstance(action, (int, np.integer)) and cfg.action_mode == "discrete":
        # 0=HOLD, 1=CHARGE, 2=DISCHARGE
        if int(action) == 1:
            return cfg.max_charge_kw
        if int(action) == 2:
            return -cfg.max_discharge_kw
        return 0.0

    # Continuous: action is power in kW (positive=charge, negative=discharge)
    try:
        power_kw = float(action)
    except Exception:
        return 0.0

    return float(np.clip(power_kw, -cfg.max_discharge_kw, cfg.max_charge_kw))

def action_to_power_kw(action: Action, cfg: BatteryEnvConfig) -> float:
    """Public wrapper for converting actions to charge/discharge power (kW)."""
    return _action_to_power_kw(action=action, cfg=cfg)


def _compute_prices(row: pd.Series, cfg: BatteryEnvConfig) -> Tuple[float, float, float]:
    base_price = float(row[cfg.base_price_col])

    if cfg.buy_price_col in row and cfg.sell_price_col in row:
        buy_price = float(row[cfg.buy_price_col])
        sell_price = float(row[cfg.sell_price_col])
        return base_price, buy_price, sell_price

    buy_price = base_price * cfg.buy_price_multiplier + cfg.buy_price_spread
    sell_price = base_price * cfg.sell_price_multiplier - cfg.sell_price_spread
    return base_price, float(buy_price), float(sell_price)

def compute_prices(row: pd.Series, cfg: BatteryEnvConfig) -> Tuple[float, float, float]:
    """Public wrapper for computing (base, buy, sell) prices for a row."""
    return _compute_prices(row=row, cfg=cfg)


def _default_household_load_kwh(hour: int, daily_kwh: float) -> float:
    # Simple deterministic shape: evening peak, midday dip.
    # Mean over 24h is daily_kwh/24.
    base = daily_kwh / 24.0
    peak = 0.8 * base * np.exp(-((hour - 19) ** 2) / (2 * (2.8**2)))
    morning = 0.5 * base * np.exp(-((hour - 8) ** 2) / (2 * (2.5**2)))
    midday_dip = -0.3 * base * np.exp(-((hour - 13) ** 2) / (2 * (3.0**2)))
    return float(max(0.05, base + peak + morning + midday_dip))


def _compute_load_kwh(row: pd.Series, hour: int, cfg: BatteryEnvConfig) -> float:
    if cfg.load_kwh_col in row:
        return float(row[cfg.load_kwh_col])

    if cfg.use_load_column_as_kwh and cfg.load_col in row:
        try:
            val = float(row[cfg.load_col])
        except Exception:
            val = float("nan")
        if np.isfinite(val) and val >= 0:
            return float(val)

    return _default_household_load_kwh(hour=hour, daily_kwh=cfg.load_profile_daily_kwh)

def compute_load_kwh(row: pd.Series, hour: int, cfg: BatteryEnvConfig) -> float:
    """Public wrapper for computing load_kwh for a row."""
    return _compute_load_kwh(row=row, hour=hour, cfg=cfg)

def transition(
    soc: float,
    power_kw: float,
    buy_price: float,
    sell_price: float,
    load_kwh: float,
    cfg: BatteryEnvConfig,
) -> Tuple[float, float, Dict[str, float]]:
    """
    One-step transition consistent with the environment.

    Args:
        soc: current SOC in [0,1]
        power_kw: positive=charge, negative=discharge (will be clipped)
        buy_price/sell_price: $/kWh
        load_kwh: household load for this step (kWh)
        cfg: env config

    Returns:
        next_soc, reward, flows
    """
    power_kw = float(np.clip(power_kw, -cfg.max_discharge_kw, cfg.max_charge_kw))
    soc = float(np.clip(soc, 0.0, 1.0))

    # Baseline (no battery): buy all load from grid.
    baseline_cost = load_kwh * buy_price

    charge_cost = 0.0
    export_revenue = 0.0
    avoided_cost = 0.0
    grid_import_kwh = 0.0
    grid_export_kwh = 0.0

    energy_kwh = soc * cfg.capacity_kwh

    if power_kw > 0 and cfg.allow_grid_charge:
        requested_from_grid = power_kw * cfg.dt_hours
        # limit by SOC headroom
        max_store_kwh = max(0.0, (cfg.max_soc * cfg.capacity_kwh) - energy_kwh)
        stored_kwh = min(max_store_kwh, requested_from_grid * cfg.eff_charge)
        actual_from_grid = 0.0 if cfg.eff_charge == 0 else stored_kwh / cfg.eff_charge

        energy_kwh += stored_kwh
        charge_cost = actual_from_grid * buy_price
        grid_import_kwh = actual_from_grid

    elif power_kw < 0:
        requested_from_batt = (-power_kw) * cfg.dt_hours
        available_kwh = max(0.0, energy_kwh - (cfg.min_soc * cfg.capacity_kwh))
        energy_from_batt = min(requested_from_batt, available_kwh)
        energy_to_bus = energy_from_batt * cfg.eff_discharge

        serve_load = min(load_kwh, energy_to_bus)
        avoided_cost = serve_load * buy_price

        if cfg.allow_export:
            grid_export_kwh = max(0.0, energy_to_bus - serve_load)
            export_revenue = grid_export_kwh * sell_price
        else:
            # If export is disallowed, don't waste discharge beyond load.
            if energy_to_bus > load_kwh and cfg.eff_discharge > 0:
                energy_from_batt = load_kwh / cfg.eff_discharge
                energy_to_bus = load_kwh

        energy_kwh -= energy_from_batt

    # HOLD (or charging disallowed): everything stays the same, reward=0 by construction.

    next_soc = float(np.clip(energy_kwh / cfg.capacity_kwh, 0.0, 1.0))
    next_soc = float(np.clip(next_soc, cfg.min_soc, cfg.max_soc))

    # Net cost with battery for incremental settlement.
    # Note: we do NOT include baseline load cost; reward is incremental vs baseline.
    reward = avoided_cost + export_revenue - charge_cost

    flows = {
        "baseline_cost": float(baseline_cost),
        "avoided_cost": float(avoided_cost),
        "charge_cost": float(charge_cost),
        "export_revenue": float(export_revenue),
        "grid_import_kwh": float(grid_import_kwh),
        "grid_export_kwh": float(grid_export_kwh),
    }
    return next_soc, float(reward), flows


class BatteryEnv(gym.Env):
    """
    Gymnasium-compatible battery arbitrage environment.

    Notes for backwards compatibility:
    - Observations are dicts with 'soc' in PERCENT [0,100] (legacy agents).
    - step() accepts both discrete actions and string actions.
    """

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, config: Optional[BatteryEnvConfig] = None):
        self.df = df.reset_index(drop=True).copy()
        if "timestamp" not in self.df.columns:
            raise ValueError("df must contain a 'timestamp' column")

        self.cfg = config or BatteryEnvConfig()

        # Expose legacy attributes
        self.capacity = self.cfg.capacity_kwh
        self.max_power = float(max(self.cfg.max_charge_kw, self.cfg.max_discharge_kw))
        self.efficiency = self.cfg.roundtrip_efficiency

        if self.cfg.action_mode == "discrete":
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(
                low=np.array([-self.cfg.max_discharge_kw], dtype=np.float32),
                high=np.array([self.cfg.max_charge_kw], dtype=np.float32),
                dtype=np.float32,
            )

        self.observation_space = spaces.Dict(
            {
                "price": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                "buy_price": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                "sell_price": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                "load_kwh": spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
                "soc": spaces.Box(low=0.0, high=100.0, shape=(), dtype=np.float32),
                "hour": spaces.Discrete(24),
                "day": spaces.Discrete(max(1, int(np.ceil(len(self.df) / 24)))),
                "step": spaces.Discrete(max(1, len(self.df))),
            }
        )

        self.current_step = 0
        self.current_soc = 0.5
        self.history: list[dict] = []
        self._np_random: Optional[np.random.Generator] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self._np_random = np.random.default_rng(seed)

        options = options or {}
        initial_soc = float(options.get("initial_soc", 0.5))
        initial_soc = float(np.clip(initial_soc, self.cfg.min_soc, self.cfg.max_soc))

        self.current_step = 0
        self.current_soc = initial_soc
        self.history = []
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> Optional[Dict[str, Any]]:
        if self.current_step >= len(self.df):
            return None

        row = self.df.iloc[self.current_step]
        timestamp = pd.to_datetime(row["timestamp"])
        hour = int(timestamp.hour)
        day = int(self.current_step // 24)

        base_price, buy_price, sell_price = _compute_prices(row=row, cfg=self.cfg)
        load_kwh = _compute_load_kwh(row=row, hour=hour, cfg=self.cfg)
        load_raw = float(row[self.cfg.load_col]) if self.cfg.load_col in row else float(load_kwh)

        return {
            "price": float(base_price),
            "buy_price": float(buy_price),
            "sell_price": float(sell_price),
            "load_kwh": float(load_kwh),
            "soc": round(self.current_soc * 100, 1),
            "hour": hour,
            "day": day,
            "step": int(self.current_step),
            "timestamp": str(timestamp),
            # Legacy key for prompts/plots
            "load": float(load_kwh),
            "load_raw": float(load_raw),
        }

    def step(
        self, action: Action
    ) -> Tuple[Optional[Dict[str, Any]], float, bool, bool, Dict[str, Any]]:
        obs = self._get_obs()
        if obs is None:
            return None, 0.0, True, False, {"error": "Episode finished"}

        row = self.df.iloc[self.current_step]
        timestamp = pd.to_datetime(row["timestamp"])

        power_kw = _action_to_power_kw(action, self.cfg)
        base_price, buy_price, sell_price = _compute_prices(row=row, cfg=self.cfg)
        load_kwh = float(obs["load_kwh"])

        soc_before = float(self.current_soc)
        next_soc, reward, flows = transition(
            soc=soc_before,
            power_kw=power_kw,
            buy_price=buy_price,
            sell_price=sell_price,
            load_kwh=load_kwh,
            cfg=self.cfg,
        )
        self.current_soc = next_soc

        info = {
            "action": "HOLD" if power_kw == 0 else ("CHARGE" if power_kw > 0 else "DISCHARGE"),
            "power_kw": float(power_kw),
            "price": float(base_price),
            "buy_price": float(buy_price),
            "sell_price": float(sell_price),
            "hour": int(timestamp.hour),
            "load": float(load_kwh),
            "load_kwh": float(load_kwh),
            "load_raw": float(obs.get("load_raw", load_kwh)),
            "soc_before": round(soc_before * 100, 1),
            "soc_after": round(self.current_soc * 100, 1),
            "grid_cost": round(flows["charge_cost"], 4),
            "grid_revenue": round(flows["export_revenue"], 4),
            "avoided_cost": round(flows["avoided_cost"], 4),
            "baseline_cost": round(flows["baseline_cost"], 4),
            "grid_import_kwh": round(flows["grid_import_kwh"], 4),
            "grid_export_kwh": round(flows["grid_export_kwh"], 4),
            "reward": round(reward, 4),
            "step": int(self.current_step),
            "day": int(self.current_step // 24),
            "timestamp": str(timestamp),
        }
        self.history.append(info)

        self.current_step += 1
        terminated = self.current_step >= len(self.df)
        truncated = False
        next_obs = self._get_obs() if not terminated else None
        return next_obs, float(reward), terminated, truncated, info
    
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
        total_avoided = sum(h.get('avoided_cost', 0) for h in day_records)
        charge_count = sum(1 for h in day_records if h['action'] == 'CHARGE')
        discharge_count = sum(1 for h in day_records if h['action'] == 'DISCHARGE')
        
        return {
            "day": day,
            "total_profit": round(total_profit, 4),
            "total_cost": round(total_cost, 4),
            "total_revenue": round(total_revenue, 4),
            "total_avoided_cost": round(total_avoided, 4),
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
    # Simple sanity check
    import os

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "caiso_enhanced_data.csv")
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    env = BatteryEnv(df)
    obs, _ = env.reset(seed=42, options={"initial_soc": 0.5})

    print("初始状态:", obs)
    print(f"总天数: {env.total_days}, 总步数: {env.total_steps}")

    for action in ["CHARGE", "DISCHARGE", "HOLD"]:
        obs, _ = env.reset(seed=42, options={"initial_soc": 0.5})
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"\n动作 {action}:")
        print(f"  奖励: ${reward:.4f}")
        print(f"  SOC: {info['soc_before']}% -> {info['soc_after']}%")
