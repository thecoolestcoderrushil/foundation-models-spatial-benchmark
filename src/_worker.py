"""
Persistent worker for per-cell timeouts.

A method registration can hang or run very long (LDDMM/GP on CPU). We run each
registration in a long-lived worker process and enforce a wall-clock timeout from
the parent: if a cell exceeds the budget, the parent terminates and restarts the
worker (paying the import cost only on a timeout, not per cell) and records a
timeout failure. Method modules are imported ONCE per worker lifetime.
"""
from __future__ import annotations

import queue
import sys
from pathlib import Path

import multiprocessing as mp

SRC = Path(__file__).resolve().parent


def _loop(inq, outq):
    sys.path.insert(0, str(SRC))
    from methods import ALL_METHODS
    reg = {c().name: c for c in ALL_METHODS}
    while True:
        item = inq.get()
        if item is None:
            return
        name, ref, mov = item
        try:
            out = reg[name]().register(ref, mov)
        except Exception as e:  # pragma: no cover
            import numpy as np
            out = dict(pred_xy=np.full((mov.n_obs, 2), float("nan")),
                       runtime_s=0.0, failed=True, reason=f"{type(e).__name__}: {e}")
        outq.put(out)


class TimeoutWorker:
    """One persistent worker; restart-on-timeout."""

    def __init__(self):
        self._start()

    def _start(self):
        ctx = mp.get_context("spawn")
        self.inq = ctx.Queue()
        self.outq = ctx.Queue()
        self.p = ctx.Process(target=_loop, args=(self.inq, self.outq), daemon=True)
        self.p.start()

    def call(self, name, ref, mov, timeout):
        import numpy as np
        self.inq.put((name, ref, mov))
        try:
            return self.outq.get(timeout=timeout), False
        except queue.Empty:
            # hung cell: kill + restart the worker, record timeout
            try:
                self.p.terminate(); self.p.join(timeout=10)
            except Exception:
                pass
            self._start()
            return (dict(pred_xy=np.full((mov.n_obs, 2), np.nan), runtime_s=timeout,
                         failed=True, reason=f"timeout >{timeout}s"), True)

    def close(self):
        try:
            self.inq.put(None); self.p.join(timeout=5)
            if self.p.is_alive():
                self.p.terminate()
        except Exception:
            pass
