# capgate: capped escalation gateway

Stage 1 of `../cs512_sprint_nov16-18.md`. Plain Python, simulated clock, no Ray yet.

    uv sync                                  # add --extra ray for Ray Serve
    uv run pytest -q
    uv run python -m capgate.run             # stand-in rows
    uv run python -m capgate.run PATH.parquet   # the UQ repo's gateway_v0.parquet

| File | What it holds |
|---|---|
| `capgate/sim.py` | the fake clock |
| `capgate/data.py` | loads the replay table, or makes stand-in rows with the same columns |
| `capgate/allocator.py` | `Lease` and `Allocator`: the cap, grants, returns, epochs |
| `capgate/replica.py` | one gateway replica: the keep-or-escalate rule and its price |
| `capgate/run.py` | bursty arrivals, N replicas, one run, the printed summary |

Not built yet: time taken by calls, crashes mid-call, the grant log on disk, allocator backup,
baselines, length predictor, audits, drift alarm, latency price, Ray Serve wrap.

A plain-words walkthrough of the whole project (problems, components, one request end to end,
fault tolerance, extensions) is in `docs/walkthrough.html`. Download it and open it in a browser.
