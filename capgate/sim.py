"""A fake clock. Events are (time, function) pairs run in time order, so a
one-hour experiment finishes in under a second."""

import heapq


class Sim:
    def __init__(self):
        self.now = 0.0
        self._queue = []
        self._count = 0  # tie-breaker: events at the same time run in the order added

    def at(self, time, fn, *args):
        heapq.heappush(self._queue, (time, self._count, fn, args))
        self._count += 1

    def run(self):
        while self._queue:
            self.now, _, fn, args = heapq.heappop(self._queue)
            fn(*args)
