"""Replay arrivals through N replicas sharing one cap, and report what happened.

    uv run python -m capgate.run [replay_table.parquet]
"""

import sys

import numpy as np

from capgate import data
from capgate.allocator import Allocator
from capgate.replica import Bill, Config, Replica
from capgate.sim import Sim


def bursty_arrivals(n, rate, seed, burst_factor=5.0, switch_prob=0.01):
    """Arrival times: steady traffic at `rate` per second, with bursts `burst_factor` times faster."""
    rng = np.random.default_rng(seed)
    times, t, burst = [], 0.0, False
    for _ in range(n):
        if rng.random() < switch_prob:
            burst = not burst
        t += rng.exponential(1.0 / (rate * (burst_factor if burst else 1.0)))
        times.append(t)
    return times


def run(table, n_requests=10_000, n_replicas=4, cap_frac=0.2, lease_len=30.0, rate=20.0,
        eta=0.05, out_quantile=0.95, seed=0, faults=None):
    """`cap_frac`: the cap as a share of what escalating every request would cost.
    `faults`: optional function(sim, replicas) that schedules faults before the run."""
    rng = np.random.default_rng(seed)
    tune, test = table[table["split"] == "dev"], table[table["split"] == "calib"]
    p_in, p_out = data.EXP_PRICE_IN / 1e6, data.EXP_PRICE_OUT / 1e6

    out_cap = int(np.quantile(tune["exp_tokens_out"], out_quantile))
    full_cost = test["exp_tokens_in"] * p_in + test["exp_tokens_out"] * p_out
    cap = cap_frac * n_requests * float(full_cost.mean())
    cfg = Config(out_cap=out_cap, price_in=p_in, price_out=p_out, eta=eta,
                 cost_ref=float(tune["exp_tokens_in"].mean()) * p_in + out_cap * p_out)

    times = bursty_arrivals(n_requests, rate, seed)
    sim, bill = Sim(), Bill()
    allocator = Allocator(cap, horizon=times[-1], lease_len=lease_len, n_replicas=n_replicas)
    replicas = [Replica(i, sim, allocator, bill, cfg) for i in range(n_replicas)]

    questions = list(test.itertuples(index=False))
    for t in times:  # each arrival: a random question sent to a random replica
        sim.at(t, replicas[rng.integers(n_replicas)].handle, questions[rng.integers(len(questions))])
    if faults:
        faults(sim, replicas)
    sim.run()
    for r in replicas:
        r.shutdown()

    log = np.array([row for r in replicas for row in r.log], dtype=float)
    return {
        "requests": len(log), "cap": cap, "spend": bill.total, "overshoot": max(0.0, bill.total - cap),
        "budget_used": bill.total / cap, "accuracy": log[:, 1].mean(), "escalated": log[:, 0].mean(),
        "flagged": log[:, 2].mean(), "all_cheap_accuracy": float(test["cheap_correct"].mean()),
        "all_expensive_accuracy": float(test["exp_correct"].mean()), "out_cap_tokens": out_cap,
    }


if __name__ == "__main__":
    table = data.load(sys.argv[1]) if len(sys.argv) > 1 else data.fake()
    print("replay table:", sys.argv[1] if len(sys.argv) > 1 else "stand-in rows (fake)")
    for key, value in run(table).items():
        print(f"  {key:24s} {value:.4f}" if isinstance(value, float) else f"  {key:24s} {value}")
