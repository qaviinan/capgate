"""The budget allocator. It owns the cap and hands it out as leases.

Safety rule: budget is counted as gone the moment it is granted. It only comes
back when the replica that holds it reports how much it actually spent. A replica
that crashes or is cut off never reports, so its unspent budget is lost, not re-granted.
That is what makes overshoot impossible: spend <= granted <= cap.
"""

from dataclasses import dataclass


@dataclass
class Lease:
    amount: float   # dollars this replica may spend
    expiry: float   # no new spending at or after this time
    epoch: int      # which allocator issued it
    spent: float = 0.0

    @property
    def left(self):
        return self.amount - self.spent


class Allocator:
    def __init__(self, cap, horizon, lease_len, n_replicas, epoch=1):
        self.cap = cap
        self.unallocated = cap
        self.horizon = horizon      # length of the budget period, seconds
        self.lease_len = lease_len
        self.n_replicas = n_replicas
        self.epoch = epoch
        self.granted = 0.0          # total ever granted, minus what was returned

    def grant(self, now, old=None):
        """Return the unspent part of `old` (if any), then issue a new lease.

        Each grant is this replica's share of an even spend rate over the time left.
        """
        self.give_back(old)
        time_left = max(self.horizon - now, self.lease_len)
        share = self.unallocated * self.lease_len / time_left / self.n_replicas
        amount = min(self.unallocated, share)
        self.unallocated -= amount
        self.granted += amount
        return Lease(amount, now + self.lease_len, self.epoch)

    def give_back(self, lease):
        if lease is None or lease.epoch != self.epoch:
            return
        self.unallocated += lease.left
        self.granted -= lease.left
        lease.amount = lease.spent  # the lease is now worth nothing more
