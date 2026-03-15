import itertools
import csv
import sys
import os
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt

import rl_traffic_agent as agent_module
from rl_traffic_agent import (
    PPOAgent, run_episode, run_baseline, NUM_EPISODES
)

NUM_CARS = 520
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please set SUMO_HOME environment variable!")

import traci


HYPERPARAMS = {
    # "GAMMA":        [0.90, 0.95, 0.99],
    "LR":           [1e-3, 1e-4, 1e-5],
    # "CLIP_EPS":     [0.1, 0.15,  0.2],
    # "PPO_EPOCHS":   [2,    4,    8],
    # "ENTROPY_COEF": [0.001, 0.01, 0.05],
    # "VALUE_COEF":   [0.25, 0.5,  1.0],
}

# Defaults as defined in rl_traffic_agent.py
DEFAULTS = {
    "GAMMA":        0.99,
    "LR":           1e-4,
    "CLIP_EPS":     0.2,
    "PPO_EPOCHS":   4,
    "ENTROPY_COEF": 0.01,
    "VALUE_COEF":   0.5,
}


def reset_defaults():
    """Restore all hyperparams to their original values."""
    for k, v in DEFAULTS.items():
        setattr(agent_module, k, v)


def train_and_eval(label):
    """Train a fresh agent and return (episode_waits, eval_wait)."""
    print(f"    Training [{label}]...")
    ppo = PPOAgent()
    ep_waits = []
    for ep in range(NUM_EPISODES):
        w = run_episode(ppo, training=True)
        ep_waits.append(w)
        print(f"      Ep {ep+1:>4}/{NUM_EPISODES} | {w:.2f}s")
    eval_wait = run_episode(ppo, training=False)
    print(f"      >> Eval: {eval_wait:.2f}s")
    return ep_waits, eval_wait


def moving_avg(data, window=10):
    return [np.mean(data[max(0, i - window + 1):i + 1]) for i in range(len(data))]


def plot_param(param, values, results, baseline_wait, timestamp):
    """
    Two-panel figure for one hyperparameter:
      Left  — learning curves (improvement % per episode)
      Right — bar chart of final eval improvement
    """
    fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(values)))

    eval_impr_b = []
    for val, color in zip(values, colors):
        ep_waits, eval_wait = results[val]
        ep_impr = [(baseline_wait - w) / baseline_wait * 100 for w in ep_waits]
        ma      = moving_avg(ep_impr, window=max(1, NUM_EPISODES // 10))

        ax_curve.plot(ep_impr, alpha=0.15, color=color)
        ax_curve.plot(ma, linewidth=2, color=color,
                      label=f"{param}={val}  (eval {(baseline_wait-eval_wait)/baseline_wait*100:+.1f}%)")
        eval_impr_b.append((baseline_wait - eval_wait) / baseline_wait * 100)

    ax_curve.axhline(30, color="red", linestyle="--", linewidth=1, label="30% target")
    ax_curve.set_xlabel("Episode"); ax_curve.set_ylabel("Improvement vs Baseline (%)")
    ax_curve.set_title(f"{param} — Learning Curves")
    ax_curve.legend(fontsize=8); ax_curve.grid(True, alpha=0.3)

    bar_labels = [str(v) for v in values]
    bars = ax_bar.bar(bar_labels, eval_impr_b, color=colors, edgecolor="white")
    ax_bar.axhline(30, color="red", linestyle="--", linewidth=1, label="30% target")

    # Highlight the default value
    default_val = DEFAULTS[param]
    for bar, val in zip(bars, values):
        if val == default_val:
            bar.set_edgecolor("black"); bar.set_linewidth(2.5)

    for bar, pct in zip(bars, eval_impr_b):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{pct:+.1f}%", ha="center", va="bottom", fontsize=9)

    ax_bar.set_xlabel(param); ax_bar.set_ylabel("Eval Improvement vs Baseline (%)")
    ax_bar.set_title(f"{param} — Final Eval (black border = default)")
    ax_bar.legend(); ax_bar.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Hyperparameter Sweep: {param}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"sweep_{param}_{timestamp}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved → {path}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = f"sweep_results_{timestamp}.csv"

    print("Running baseline...")
    baseline_wait = run_baseline()
    print(f"Baseline: {baseline_wait:.2f}s ({baseline_wait/NUM_CARS:.2f}s/car)\n")

    # print("Running Webster...")
    # webster_wait = run_webster()
    # print(f"Webster:  {webster_wait:.2f}s ({webster_wait/NUM_CARS:.2f}s/car)\n")

    csv_rows = []

    for param, values in HYPERPARAMS.items():
        print(f"\n{'='*60}")
        print(f"Sweeping: {param}  (values: {values}, default: {DEFAULTS[param]})")
        print(f"{'='*60}")

        param_results = {}
        for val in values:
            reset_defaults()
            setattr(agent_module, param, val)

            ep_waits, eval_wait = train_and_eval(f"{param}={val}")
            param_results[val] = (ep_waits, eval_wait)

            pct_b = (baseline_wait - eval_wait) / baseline_wait * 100
            # pct_w = (webster_wait  - eval_wait) / webster_wait  * 100
            csv_rows.append({
                "param": param, "value": val,
                "eval_wait": round(eval_wait, 2),
                "pct_vs_baseline": round(pct_b, 2),
                # "pct_vs_webster":  round(pct_w, 2),
                "is_default": val == DEFAULTS[param],
            })

        plot_param(param, values, param_results, baseline_wait, timestamp)

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["param", "value", "eval_wait",
                                               "pct_vs_baseline", "is_default"])
        writer.writeheader(); writer.writerows(csv_rows)
    print(f"\nResults saved → {csv_path}")

    # Summary leaderboard
    print(f"\n{'='*60}\nSUMMARY — Best value per hyperparameter\n{'='*60}")
    for param in HYPERPARAMS:
        rows = [r for r in csv_rows if r["param"] == param]
        best = max(rows, key=lambda r: r["pct_vs_baseline"])
        dfl  = next(r for r in rows if r["is_default"])
        print(f"  {param:<14} best={best['value']} ({best['pct_vs_baseline']:+.1f}%)  "
              f"default={dfl['value']} ({dfl['pct_vs_baseline']:+.1f}%)")

    reset_defaults()


if __name__ == "__main__":
    main()
