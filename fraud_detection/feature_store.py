"""In-memory card-history store: the online counterpart of ``features.card_history``.

In production the per-card history (velocity counts, recent spend, time since
the previous transaction, average amount) comes from a feature store that is
updated as transactions stream in. This module is a minimal in-memory version:

    store = CardHistoryStore()
    store.warm_up(past_transactions)          # optional: load earlier history
    hist = store.get(cc_num, timestamp)       # history BEFORE this transaction
    ...score the transaction with hist...
    store.update(cc_num, timestamp, amount)   # then record it

It produces exactly the same values as the batch feature code used for
training (``tests/test_feature_store.py`` checks this), so the model sees the
same features online as it did offline.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import NO_HISTORY_SECONDS

WINDOWS = {"1h": 3600, "24h": 86400, "7d": 7 * 86400}
MAX_WINDOW = max(WINDOWS.values())


def to_epoch_seconds(ts) -> int:
    """Seconds since epoch for a timestamp-like value (unit-safe across pandas versions)."""
    return int(np.datetime64(pd.Timestamp(ts).to_datetime64(), "s").astype(np.int64))


@dataclass
class _CardState:
    recent: deque = field(default_factory=deque)  # (epoch_seconds, amount) within the last 7 days
    n_total: int = 0
    sum_total: float = 0.0
    last_ts: int | None = None


class CardHistoryStore:
    def __init__(self) -> None:
        self._cards: dict[int, _CardState] = {}

    def __len__(self) -> int:
        return len(self._cards)

    def get(self, cc_num: int, ts) -> dict:
        """History for a transaction at ``ts``, using only the card's earlier transactions."""
        t = to_epoch_seconds(ts)
        state = self._cards.get(int(cc_num))
        if state is None or state.n_total == 0:
            return {"secs_since_last_txn": float(NO_HISTORY_SECONDS), "card_txn_count_1h": 0,
                    "card_txn_count_24h": 0, "card_txn_count_7d": 0, "card_amt_sum_24h": 0.0,
                    "card_avg_amt": None}
        while state.recent and state.recent[0][0] < t - MAX_WINDOW:
            state.recent.popleft()
        counts = {name: 0 for name in WINDOWS}
        sum_24h = 0.0
        for ts_prev, amt_prev in state.recent:
            for name, window in WINDOWS.items():
                if ts_prev >= t - window:
                    counts[name] += 1
            if ts_prev >= t - WINDOWS["24h"]:
                sum_24h += amt_prev
        return {
            "secs_since_last_txn": float(t - state.last_ts),
            "card_txn_count_1h": counts["1h"],
            "card_txn_count_24h": counts["24h"],
            "card_txn_count_7d": counts["7d"],
            "card_amt_sum_24h": float(sum_24h),
            "card_avg_amt": state.sum_total / state.n_total,
        }

    def update(self, cc_num: int, ts, amount: float) -> None:
        """Record a transaction (call after scoring it)."""
        t = to_epoch_seconds(ts)
        state = self._cards.setdefault(int(cc_num), _CardState())
        state.recent.append((t, float(amount)))
        state.n_total += 1
        state.sum_total += float(amount)
        state.last_ts = t

    def warm_up(self, transactions: pd.DataFrame) -> None:
        """Load history from past transactions (columns: cc_num, trans_date_trans_time, amt).

        Totals and the last timestamp come from every row; only the last 7 days
        before the newest transaction are kept for the window counts.
        """
        df = transactions[["cc_num", "trans_date_trans_time", "amt"]].copy()
        df["t"] = pd.to_datetime(df["trans_date_trans_time"]).to_numpy().astype("datetime64[s]").astype(np.int64)
        df = df.sort_values("t", kind="stable")
        agg = df.groupby("cc_num").agg(n=("amt", "size"), s=("amt", "sum"), last=("t", "max"))
        recent = df[df["t"] >= df["t"].max() - MAX_WINDOW]
        recent_by_card = {cc: list(zip(g["t"].tolist(), g["amt"].tolist())) for cc, g in recent.groupby("cc_num")}
        for cc, row in agg.iterrows():
            state = self._cards.setdefault(int(cc), _CardState())
            state.n_total += int(row["n"])
            state.sum_total += float(row["s"])
            state.last_ts = int(row["last"]) if state.last_ts is None else max(state.last_ts, int(row["last"]))
            state.recent.extend(recent_by_card.get(cc, []))
