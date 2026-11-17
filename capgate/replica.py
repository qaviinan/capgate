"""One gateway replica. It decides alone, per request, whether to keep the cheap
answer or pay for the expensive one, using only its own lease and its own price."""

import math
from dataclasses import dataclass


@dataclass
class Config:
    out_cap: int          # max output tokens allowed on an escalation
    price_in: float       # dollars per input token, expensive tier
    price_out: float      # dollars per output token, expensive tier
    cost_ref: float       # a typical escalation cost; keeps the price `lam` unit-free
    lam0: float = 0.5     # starting price
    eta: float = 0.05     # how fast the price reacts; 0 keeps it fixed
    lam_min: float = 0.01
    lam_max: float = 50.0


class Bill:
    """What the provider actually charged, summed over all replicas. Ground truth."""

    def __init__(self):
        self.total = 0.0


class Replica:
    def __init__(self, rid, sim, allocator, bill, cfg):
        self.rid, self.sim, self.allocator, self.bill, self.cfg = rid, sim, allocator, bill, cfg
        self.lease = None
        self.lam = cfg.lam0
        self.last_seen = 0.0
        self.cut_off = False   # fault switch: allocator unreachable
        self.log = []          # one (escalated, correct, flagged, cost) per request

    def _renew(self):
        if self.cut_off:
            return
        lease = self.allocator.grant(self.sim.now, self.lease)
        if self.lease is None or lease.epoch >= self.lease.epoch:  # never accept an older allocator
            self.lease = lease

    def handle(self, q):
        now = self.sim.now
        if self.lease is None or now >= self.lease.expiry:
            self._renew()
        valid = self.lease is not None and now < self.lease.expiry

        # Upper bound on what escalating could cost: real input, capped output.
        cost_bound = q.exp_tokens_in * self.cfg.price_in + self.cfg.out_cap * self.cfg.price_out
        gain = 1.0 - q.score   # chance the cheap answer is wrong, if the score is right
        wanted = valid and gain > self.lam * cost_bound / self.cfg.cost_ref
        escalate = wanted and self.lease.left >= cost_bound

        cost = 0.0
        if escalate:
            out = min(q.exp_tokens_out, self.cfg.out_cap)          # max_tokens cuts long answers
            cost = q.exp_tokens_in * self.cfg.price_in + out * self.cfg.price_out
            self.lease.spent += cost
            self.bill.total += cost
            cut = q.exp_tokens_out > self.cfg.out_cap
            correct = 0 if cut else q.exp_correct                 # a cut-off answer counts as wrong
        else:
            correct = q.cheap_correct
        self.log.append((escalate, correct, not valid, cost))

        if valid and self.cfg.eta > 0:
            # Price on what we wanted to spend, so an empty lease pushes the price up, not down.
            self._update_price(cost if escalate else cost_bound * wanted, now - self.last_seen)
        self.last_seen = now

    def _update_price(self, cost, dt):
        """Raise the price when spending runs ahead of an even pace over the lease, lower it when behind."""
        on_pace = self.lease.amount / self.allocator.lease_len * dt
        step = self.cfg.eta * (cost - on_pace) / self.cfg.cost_ref
        self.lam = min(self.cfg.lam_max, max(self.cfg.lam_min, self.lam * math.exp(step)))

    def shutdown(self):
        if not self.cut_off:
            self.allocator.give_back(self.lease)
