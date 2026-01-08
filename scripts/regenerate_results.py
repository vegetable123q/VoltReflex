"""
Regenerate experiment results (14 days) and update output artifacts.

This script is meant to be the single source of truth for the README results.
Experiment settings / running instructions live in startup.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import json
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUTS = ROOT / "outputs"

from src.agents import ReflexionAgent, RuleAgent, SimpleLLMAgent
from src.env import BatteryEnv, BatteryEnvConfig
from src.rl_baselines import DQNAgent, MPCBaseline, SimpleQAgent
from src.utils import load_market_data


def _summary_to_markdown_table(summary: pd.DataFrame) -> str:
    medals = ["🥇", "🥈", "🥉"]
    lines = []
    lines.append("| 排名 | 方法 | 总利润($) | 充电 | 放电 | 持有 | LLM调用 | 相对MPC |")
    lines.append("|:----:|------|--------:|-----:|-----:|-----:|--------:|--------:|")

    for idx, row in summary.iterrows():
        rank = medals[idx] if idx < 3 else str(idx + 1)
        method = str(row["Method"])
        profit = float(row["Profit"])
        rel = float(row["Relative_to_MPC"])
        lines.append(
            f"| {rank} | {method} | {profit:.2f} | {int(row['Charge'])} | {int(row['Discharge'])} | {int(row['Hold'])} | {int(row['LLM_Calls'])} | {rel:.1f}% |"
        )
    return "\n".join(lines)


def _update_readme_results_table(readme_path: Path, summary: pd.DataFrame) -> None:
    start = "<!-- RESULTS_TABLE_START -->"
    end = "<!-- RESULTS_TABLE_END -->"
    text = readme_path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        print(f"⚠️  Skipping README update: markers not found in {readme_path}", flush=True)
        return

    table = _summary_to_markdown_table(summary)
    before, rest = text.split(start, 1)
    _mid, after = rest.split(end, 1)
    updated = before + start + "\n" + table + "\n" + end + after
    readme_path.write_text(updated, encoding="utf-8")
    print(f"✅ Updated README table: {readme_path}", flush=True)


def _cache_path(method: str) -> Path:
    safe = method.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    return OUTPUTS / f"_regen_cache_{safe}.json"


def _save_cached_run(method: str, meta: dict, payload: dict) -> None:
    path = _cache_path(method)
    data = {"meta": meta, "payload": payload}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _load_cached_run(method: str, meta: dict) -> Optional[dict]:
    path = _cache_path(method)
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if obj.get("meta") != meta:
        return None
    return obj.get("payload")


def _load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _make_env_config(cfg: dict) -> BatteryEnvConfig:
    battery_cfg = cfg.get("battery", {})
    max_power = float(battery_cfg.get("max_power_kw", 5.0))
    return BatteryEnvConfig(
        capacity_kwh=float(battery_cfg.get("capacity_kwh", 13.5)),
        max_charge_kw=max_power,
        max_discharge_kw=max_power,
        roundtrip_efficiency=float(battery_cfg.get("efficiency", 0.9)),
        min_soc=float(battery_cfg.get("min_soc", 0.10)),
        max_soc=float(battery_cfg.get("max_soc", 0.95)),
        allow_export=True,
        allow_grid_charge=True,
        action_mode="discrete",
    )


def _run_agent(
    agent,
    df: pd.DataFrame,
    *,
    days: int,
    env_cfg: BatteryEnvConfig,
    seed: int = 42,
    initial_soc: float = 0.5,
) -> Tuple[float, List[float], List[float], List[Dict], int, Dict[str, int]]:
    env = BatteryEnv(df.copy(), config=env_cfg)
    agent.reset()
    obs, _ = env.reset(seed=seed, options={"initial_soc": initial_soc})

    hourly: List[float] = []
    history: List[Dict] = []

    for day in range(days):
        print(f"  Day {day + 1}/{days}", flush=True)
        daily_buffer = []
        for _hour in range(24):
            if obs is None:
                break
            action = agent.decide(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            hourly.append(float(reward))
            daily_buffer.append(info)
            history.append(info)
            if hasattr(agent, "record_transaction"):
                agent.record_transaction(info)
            obs = next_obs
            if done:
                break
        agent.end_of_day(daily_buffer)

    daily = [sum(hourly[i : i + 24]) for i in range(0, len(hourly), 24)]
    counts = {
        "CHARGE": sum(1 for h in history if h.get("action") == "CHARGE"),
        "DISCHARGE": sum(1 for h in history if h.get("action") == "DISCHARGE"),
        "HOLD": sum(1 for h in history if h.get("action") == "HOLD"),
    }
    llm_calls = int(getattr(agent, "total_llm_calls", 0))
    return float(sum(hourly)), hourly, daily, history, llm_calls, counts


def _run_mpc(
    df: pd.DataFrame,
    *,
    days: int,
    env_cfg: BatteryEnvConfig,
    horizon: int = 24,
    seed: int = 42,
    initial_soc: float = 0.5,
) -> Tuple[float, List[float], List[float], List[Dict], Dict[str, int]]:
    env = BatteryEnv(df.copy(), config=env_cfg)
    obs, _ = env.reset(seed=seed, options={"initial_soc": initial_soc})

    mpc = MPCBaseline(horizon=horizon)
    mpc.set_price_forecast(df[env_cfg.base_price_col].tolist())

    hourly: List[float] = []
    history: List[Dict] = []

    for _ in range(days * 24):
        if obs is None:
            break
        action = mpc.decide(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        hourly.append(float(reward))
        history.append(info)
        obs = next_obs
        if done:
            break

    daily = [sum(hourly[i : i + 24]) for i in range(0, len(hourly), 24)]
    counts = {
        "CHARGE": sum(1 for h in history if h.get("action") == "CHARGE"),
        "DISCHARGE": sum(1 for h in history if h.get("action") == "DISCHARGE"),
        "HOLD": sum(1 for h in history if h.get("action") == "HOLD"),
    }
    return float(sum(hourly)), hourly, daily, history, counts


def _run_q_learning(
    df: pd.DataFrame,
    *,
    days: int,
    env_cfg: BatteryEnvConfig,
    n_episodes: int = 100,
    seed: int = 42,
) -> Tuple[float, List[float], List[float], List[Dict], Dict[str, int]]:
    np.random.seed(seed)
    agent = SimpleQAgent()
    for ep in range(n_episodes):
        env = BatteryEnv(df.copy(), config=env_cfg)
        agent.train_episode(env)
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}", flush=True)

    eval_env = BatteryEnv(df.copy(), config=env_cfg)
    total, history = agent.evaluate(eval_env)
    hourly = [float(h["reward"]) for h in history]
    daily = [sum(hourly[i : i + 24]) for i in range(0, len(hourly), 24)]
    counts = {
        "CHARGE": sum(1 for h in history if h.get("action") == "CHARGE"),
        "DISCHARGE": sum(1 for h in history if h.get("action") == "DISCHARGE"),
        "HOLD": sum(1 for h in history if h.get("action") == "HOLD"),
    }
    return float(total), hourly, daily, history, counts


def _run_dqn(
    df: pd.DataFrame,
    *,
    days: int,
    env_cfg: BatteryEnvConfig,
    n_episodes: int = 100,
    seed: int = 42,
) -> Tuple[float, List[float], List[float], List[Dict], Dict[str, int]]:
    np.random.seed(seed)
    agent = DQNAgent()
    for ep in range(n_episodes):
        env = BatteryEnv(df.copy(), config=env_cfg)
        agent.train_episode(env)
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}", flush=True)

    eval_env = BatteryEnv(df.copy(), config=env_cfg)
    total, history = agent.evaluate(eval_env)
    hourly = [float(h["reward"]) for h in history]
    daily = [sum(hourly[i : i + 24]) for i in range(0, len(hourly), 24)]
    counts = {
        "CHARGE": sum(1 for h in history if h.get("action") == "CHARGE"),
        "DISCHARGE": sum(1 for h in history if h.get("action") == "DISCHARGE"),
        "HOLD": sum(1 for h in history if h.get("action") == "HOLD"),
    }
    return float(total), hourly, daily, history, counts


def _plot_full_comparison(
    summary: pd.DataFrame, *, out_path: Path, title: str = "14-Day Comparison"
):
    methods = summary["Method"].tolist()
    profits = summary["Profit"].to_numpy()
    charge = summary["Charge"].to_numpy()
    discharge = summary["Discharge"].to_numpy()
    hold = summary["Hold"].to_numpy()

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.barh(methods, profits, color="#4C78A8")
    ax1.set_xlabel("Total Profit ($)")
    ax1.set_title("Total Profit")
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.invert_yaxis()

    total = charge + discharge + hold
    charge_r = charge / total
    discharge_r = discharge / total
    hold_r = hold / total

    ax2.barh(methods, hold_r, label="HOLD", color="#9E9E9E")
    ax2.barh(methods, charge_r, left=hold_r, label="CHARGE", color="#2ECC71")
    ax2.barh(
        methods,
        discharge_r,
        left=hold_r + charge_r,
        label="DISCHARGE",
        color="#E74C3C",
    )
    ax2.set_xlabel("Action Ratio")
    ax2.set_xlim(0, 1)
    ax2.set_title("Action Distribution")
    ax2.invert_yaxis()
    ax2.legend(loc="lower right", frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_cumulative(hourly_by_method: Dict[str, List[float]], out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, hourly in hourly_by_method.items():
        ax.plot(np.cumsum(hourly), label=method)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Cumulative Profit ($)")
    ax.set_title("Cumulative Profit (14 Days)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_daily(daily_by_method: Dict[str, List[float]], out_path: Path):
    df = []
    for method, daily in daily_by_method.items():
        for i, v in enumerate(daily):
            df.append({"Method": method, "Day": i + 1, "Profit": v})
    plot_df = pd.DataFrame(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    for method in plot_df["Method"].unique():
        s = plot_df[plot_df["Method"] == method].sort_values("Day")
        ax.plot(s["Day"], s["Profit"], marker="o", linewidth=1.5, label=method)
    ax.set_xlabel("Day")
    ax.set_ylabel("Daily Profit ($)")
    ax.set_title("Daily Profit (14 Days)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=100, help="RL training episodes for Q/DQN")
    parser.add_argument("--skip-llm", action="store_true", help="Skip Simple LLM / Reflexion runs")
    parser.add_argument("--skip-rl", action="store_true", help="Skip Q-Learning / DQN runs")
    parser.add_argument("--no-plots", action="store_true", help="Do not regenerate PNG plots")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Ignore caches and recompute everything")
    args = parser.parse_args()

    load_dotenv(dotenv_path=ROOT / ".env")
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(ROOT / "configs" / "default.yaml")
    days = int(args.days or cfg.get("experiment", {}).get("num_days", 14))
    seed = int(args.seed or cfg.get("experiment", {}).get("seed", 42))
    initial_soc = float(cfg.get("battery", {}).get("initial_soc", 0.5))

    env_cfg = _make_env_config(cfg)

    df = load_market_data(cfg.get("data", {}).get("path"))
    df = df.head(days * 24).copy()
    print(f"📊 Data hours: {len(df)} (days={days})", flush=True)

    data_fingerprint = {
        "path": str(cfg.get("data", {}).get("path")),
        "rows": int(len(df)),
        "ts0": str(df["timestamp"].iloc[0]),
        "ts1": str(df["timestamp"].iloc[-1]),
        "p0": float(df[env_cfg.base_price_col].iloc[0]),
        "p1": float(df[env_cfg.base_price_col].iloc[-1]),
    }
    cache_meta = {
        "days": days,
        "seed": seed,
        "initial_soc": initial_soc,
        "episodes": int(args.episodes),
        "env_config": asdict(env_cfg),
        "data": data_fingerprint,
    }

    # Agents (use config where available)
    rule_cfg = cfg.get("agents", {}).get("rule", {})
    rule = RuleAgent(
        charge_threshold=float(rule_cfg.get("charge_threshold", 0.03)),
        discharge_threshold=float(rule_cfg.get("discharge_threshold", 0.08)),
        max_soc=float(rule_cfg.get("max_soc_for_charge", 0.90) * 100),
        min_soc=float(rule_cfg.get("min_soc_for_discharge", 0.15) * 100),
    )

    results = {}
    hourly = {}
    daily = {}

    # 1) MPC (reference)
    print("▶ Running: MPC (24h)", flush=True)
    cached = None if args.force else _load_cached_run("MPC (24h)", cache_meta)
    if cached:
        mpc_profit = float(cached["profit"])
        mpc_hourly = list(cached["hourly"])
        mpc_daily = list(cached["daily"])
        mpc_counts = dict(cached["counts"])
    else:
        mpc_profit, mpc_hourly, mpc_daily, _mpc_hist, mpc_counts = _run_mpc(
            df, days=days, env_cfg=env_cfg, horizon=24, seed=seed, initial_soc=initial_soc
        )
        _save_cached_run(
            "MPC (24h)",
            cache_meta,
            {"profit": mpc_profit, "hourly": mpc_hourly, "daily": mpc_daily, "counts": mpc_counts},
        )
    results["MPC (24h)"] = (mpc_profit, 0, mpc_counts)
    hourly["MPC (24h)"] = mpc_hourly
    daily["MPC (24h)"] = mpc_daily

    # 2) Rule
    print("▶ Running: Rule-Based", flush=True)
    cached = None if args.force else _load_cached_run("Rule-Based", cache_meta)
    if cached:
        profit = float(cached["profit"])
        h = list(cached["hourly"])
        d = list(cached["daily"])
        llm_calls = int(cached["llm_calls"])
        counts = dict(cached["counts"])
    else:
        profit, h, d, _hist, llm_calls, counts = _run_agent(
            rule, df, days=days, env_cfg=env_cfg, seed=seed, initial_soc=initial_soc
        )
        _save_cached_run(
            "Rule-Based",
            cache_meta,
            {"profit": profit, "hourly": h, "daily": d, "llm_calls": llm_calls, "counts": counts},
        )
    results["Rule-Based"] = (profit, llm_calls, counts)
    hourly["Rule-Based"] = h
    daily["Rule-Based"] = d

    # 3) LLM agents (optional)
    if (not args.skip_llm) and os.getenv("OPENAI_API_KEY"):
        print("▶ Running: Simple LLM (may take a while)", flush=True)
        cached = None if args.force else _load_cached_run("Simple LLM", cache_meta)
        if cached:
            profit = float(cached["profit"])
            h = list(cached["hourly"])
            d = list(cached["daily"])
            llm_calls = int(cached["llm_calls"])
            counts = dict(cached["counts"])
        else:
            llm = SimpleLLMAgent()
            profit, h, d, _hist, llm_calls, counts = _run_agent(
                llm, df, days=days, env_cfg=env_cfg, seed=seed, initial_soc=initial_soc
            )
            _save_cached_run(
                "Simple LLM",
                cache_meta,
                {"profit": profit, "hourly": h, "daily": d, "llm_calls": llm_calls, "counts": counts},
            )
        results["Simple LLM"] = (profit, llm_calls, counts)
        hourly["Simple LLM"] = h
        daily["Simple LLM"] = d

        print("▶ Running: Reflexion (may take a while)", flush=True)
        cached = None if args.force else _load_cached_run("Reflexion", cache_meta)
        if cached:
            profit = float(cached["profit"])
            h = list(cached["hourly"])
            d = list(cached["daily"])
            llm_calls = int(cached["llm_calls"])
            counts = dict(cached["counts"])
        else:
            reflexion = ReflexionAgent()
            profit, h, d, _hist, llm_calls, counts = _run_agent(
                reflexion, df, days=days, env_cfg=env_cfg, seed=seed, initial_soc=initial_soc
            )
            _save_cached_run(
                "Reflexion",
                cache_meta,
                {"profit": profit, "hourly": h, "daily": d, "llm_calls": llm_calls, "counts": counts},
            )
        results["Reflexion"] = (profit, llm_calls, counts)
        hourly["Reflexion"] = h
        daily["Reflexion"] = d
    elif args.skip_llm:
        print("⏭️  Skipping LLM agents (--skip-llm)", flush=True)
    else:
        print("⏭️  Skipping LLM agents (OPENAI_API_KEY not set)", flush=True)

    # 4) RL baselines
    if not args.skip_rl:
        print(f"▶ Training: Q-Learning (episodes={args.episodes})", flush=True)
        cached = None if args.force else _load_cached_run("Q-Learning", cache_meta)
        if cached:
            q_profit = float(cached["profit"])
            q_hourly = list(cached["hourly"])
            q_daily = list(cached["daily"])
            q_counts = dict(cached["counts"])
        else:
            q_profit, q_hourly, q_daily, _q_hist, q_counts = _run_q_learning(
                df, days=days, env_cfg=env_cfg, n_episodes=int(args.episodes), seed=seed
            )
            _save_cached_run(
                "Q-Learning",
                cache_meta,
                {"profit": q_profit, "hourly": q_hourly, "daily": q_daily, "counts": q_counts},
            )
        results["Q-Learning"] = (q_profit, 0, q_counts)
        hourly["Q-Learning"] = q_hourly
        daily["Q-Learning"] = q_daily

        print(f"▶ Training: DQN (episodes={args.episodes})", flush=True)
        cached = None if args.force else _load_cached_run("DQN", cache_meta)
        if cached:
            dqn_profit = float(cached["profit"])
            dqn_hourly = list(cached["hourly"])
            dqn_daily = list(cached["daily"])
            dqn_counts = dict(cached["counts"])
        else:
            dqn_profit, dqn_hourly, dqn_daily, _dqn_hist, dqn_counts = _run_dqn(
                df, days=days, env_cfg=env_cfg, n_episodes=int(args.episodes), seed=seed
            )
            _save_cached_run(
                "DQN",
                cache_meta,
                {"profit": dqn_profit, "hourly": dqn_hourly, "daily": dqn_daily, "counts": dqn_counts},
            )
        results["DQN"] = (dqn_profit, 0, dqn_counts)
        hourly["DQN"] = dqn_hourly
        daily["DQN"] = dqn_daily
    else:
        print("⏭️  Skipping RL baselines (--skip-rl)", flush=True)

    # Summary table
    rows = []
    for method, (profit, llm_calls, counts) in results.items():
        rows.append(
            {
                "Method": method,
                "Profit": float(profit),
                "Charge": int(counts["CHARGE"]),
                "Discharge": int(counts["DISCHARGE"]),
                "Hold": int(counts["HOLD"]),
                "LLM_Calls": int(llm_calls),
            }
        )

    summary = pd.DataFrame(rows)
    mpc_ref = float(summary.loc[summary["Method"] == "MPC (24h)", "Profit"].iloc[0])
    summary["Relative_to_MPC"] = summary["Profit"] / (mpc_ref if mpc_ref != 0 else np.nan) * 100.0

    # Rank by profit (desc)
    summary = summary.sort_values("Profit", ascending=False).reset_index(drop=True)

    out_csv = OUTPUTS / "full_experiment_results_14days.csv"
    summary.to_csv(out_csv, index=False)

    _update_readme_results_table(ROOT / "README.md", summary)

    if not args.no_plots:
        _plot_full_comparison(summary, out_path=OUTPUTS / "full_comparison_chart.png")
        _plot_cumulative(hourly, out_path=OUTPUTS / "cumulative_profits.png")
        _plot_daily(daily, out_path=OUTPUTS / "daily_profits.png")

        # Action distribution (stacked bar)
        fig, ax = plt.subplots(figsize=(10, 5))
        plot = summary.copy()
        plot = plot.set_index("Method")
        total = plot["Charge"] + plot["Discharge"] + plot["Hold"]
        ax.bar(plot.index, plot["Hold"] / total, label="HOLD", color="#9E9E9E")
        ax.bar(
            plot.index,
            plot["Charge"] / total,
            bottom=plot["Hold"] / total,
            label="CHARGE",
            color="#2ECC71",
        )
        ax.bar(
            plot.index,
            plot["Discharge"] / total,
            bottom=(plot["Hold"] + plot["Charge"]) / total,
            label="DISCHARGE",
            color="#E74C3C",
        )
        ax.set_ylabel("Action Ratio")
        ax.set_title("Action Distribution (14 Days)")
        ax.legend(frameon=False)
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(OUTPUTS / "action_distribution.png", dpi=200)
        plt.close(fig)
    else:
        print("⏭️  Skipping plots (--no-plots)", flush=True)

    meta = {
        "days": days,
        "seed": seed,
        "initial_soc": initial_soc,
        "env_config": asdict(env_cfg),
    }
    (OUTPUTS / "results_metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    print(f"✅ Wrote: {out_csv}")
    print(f"✅ Updated plots under: {OUTPUTS}")


if __name__ == "__main__":
    main()
