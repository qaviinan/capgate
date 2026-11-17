import pytest

from capgate import data
from capgate.allocator import Allocator
from capgate.run import run

TABLE = data.fake(seed=1)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("cap_frac", [0.02, 0.2, 0.6])
def test_spend_never_exceeds_cap(seed, cap_frac):
    r = run(TABLE, n_requests=3000, cap_frac=cap_frac, seed=seed)
    assert r["spend"] <= r["cap"]
    assert r["requests"] == 3000


def test_escalating_beats_all_cheap():
    r = run(TABLE, n_requests=5000, cap_frac=0.3)
    assert r["accuracy"] > r["all_cheap_accuracy"]


def test_cut_off_replica_fails_closed():
    """Cut every replica off at t=0: nobody can get a lease, so nothing is spent and all requests are flagged."""
    def faults(sim, replicas):
        for r in replicas:
            r.cut_off = True
    r = run(TABLE, n_requests=1000, faults=faults)
    assert r["spend"] == 0 and r["flagged"] == 1.0


def test_unreported_lease_is_not_regranted():
    a = Allocator(cap=100.0, horizon=100.0, lease_len=10.0, n_replicas=2)
    lost = a.grant(now=0.0)             # this replica crashes and never reports
    kept = a.grant(now=0.0)
    kept.spent = 1.0
    a.give_back(kept)
    assert a.unallocated == pytest.approx(100.0 - lost.amount - 1.0)


def test_old_epoch_cannot_return_budget():
    a = Allocator(cap=100.0, horizon=100.0, lease_len=10.0, n_replicas=1, epoch=2)
    stale = Allocator(cap=100.0, horizon=100.0, lease_len=10.0, n_replicas=1, epoch=1).grant(now=0.0)
    a.give_back(stale)
    assert a.unallocated == 100.0
