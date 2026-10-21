"""Real-time demo: stream transactions to the running API, the way a payment system would.

    make serve                                   # terminal 1
    make replay                                  # terminal 2 (defaults below)
    python -m fraud_detection.replay --limit 5000 --start "2020-06-22" --batch-size 1

For each transaction, in time order:
  1. look up the card's history in an in-memory feature store (CardHistoryStore),
  2. POST the transaction + history to /predict (or /predict/batch),
  3. record the transaction in the store, and print an alert if it was flagged.

Before streaming, the store is warmed up with earlier transactions (by default
fraudTrain.csv plus any rows of the replay file before --start), so cards
start with their real history. If the file has an ``is_fraud`` column, the
replay also reports running precision and recall.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from . import config
from .feature_store import CardHistoryStore

TRANSACTION_FIELDS = ["trans_date_trans_time", "amt", "category", "merch_lat", "merch_long",
                      "gender", "dob", "lat", "long", "city_pop"]
READ_COLUMNS = TRANSACTION_FIELDS + ["cc_num", "is_fraud"]

Scorer = Callable[[list[dict]], list[dict]]


def http_scorer(api_url: str, timeout: float = 10.0) -> Scorer:
    """Send payloads to the API: one -> /predict, several -> /predict/batch."""
    base = api_url.rstrip("/")

    def post(path: str, body) -> object:
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def score(payloads: list[dict]) -> list[dict]:
        if len(payloads) == 1:
            return [post("/predict", payloads[0])]
        return post("/predict/batch", {"transactions": payloads})

    return score


def to_payload(row: dict, history: dict) -> dict:
    payload = {k: row[k] for k in TRANSACTION_FIELDS}
    payload["trans_date_trans_time"] = pd.Timestamp(row["trans_date_trans_time"]).isoformat()
    payload["dob"] = str(pd.Timestamp(row["dob"]).date())
    payload["city_pop"] = int(payload["city_pop"])
    payload.update(history)
    return payload


def replay(rows: Iterable[dict], store: CardHistoryStore, score: Scorer, batch_size: int = 1,
           show_alerts: bool = True, report_every: int = 500) -> dict:
    """Stream ``rows`` (dicts in the Kaggle schema, time-ordered) through ``score``."""
    stats = {"scored": 0, "alerts": 0, "tp": 0, "fp": 0, "fn": 0, "labelled": False}
    latencies: list[float] = []
    results: list[dict] = []
    pending_rows: list[dict] = []
    pending_payloads: list[dict] = []

    def flush() -> None:
        if not pending_payloads:
            return
        t0 = time.perf_counter()
        preds = score(pending_payloads)
        latencies.append((time.perf_counter() - t0) * 1000 / len(pending_payloads))
        for row, pred in zip(pending_rows, preds):
            stats["scored"] += 1
            label = row.get("is_fraud")
            labelled = label is not None and not (isinstance(label, float) and math.isnan(label))
            stats["labelled"] |= labelled
            if pred["is_fraud"]:
                stats["alerts"] += 1
                if labelled:
                    stats["tp" if label == 1 else "fp"] += 1
                if show_alerts:
                    verdict = "" if not labelled else ("  [TRUE FRAUD]" if label == 1 else "  [false alarm]")
                    print(f"ALERT {row['trans_date_trans_time']}  card ...{str(row['cc_num'])[-4:]}  "
                          f"${row['amt']:>9,.2f}  {row['category']:<15} p={pred['fraud_probability']:.3f} "
                          f"{pred['risk_level']}{verdict}")
            elif labelled and label == 1:
                stats["fn"] += 1
            results.append({"trans_date_trans_time": row["trans_date_trans_time"], "cc_num": row["cc_num"],
                            "fraud_probability": pred["fraud_probability"], "is_fraud_pred": pred["is_fraud"],
                            "is_fraud": row.get("is_fraud")})
            if report_every and stats["scored"] % report_every == 0:
                print(f"... {stats['scored']:,} scored, {stats['alerts']} alerts{_running(stats)}")
        pending_rows.clear()
        pending_payloads.clear()

    for row in rows:
        history = store.get(row["cc_num"], row["trans_date_trans_time"])
        pending_rows.append(row)
        pending_payloads.append(to_payload(row, history))
        store.update(row["cc_num"], row["trans_date_trans_time"], row["amt"])
        if len(pending_payloads) >= batch_size:
            flush()
    flush()

    stats["latency_ms_p50"] = statistics.median(latencies) if latencies else None
    stats["latency_ms_p95"] = sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else None
    stats["results"] = results
    return stats


def _running(s: dict) -> str:
    if not s["labelled"]:
        return ""
    p = f"{s['tp'] / (s['tp'] + s['fp']):.3f}" if s["tp"] + s["fp"] else "n/a"
    r = f"{s['tp'] / (s['tp'] + s['fn']):.3f}" if s["tp"] + s["fn"] else "n/a (no fraud yet)"
    return f" | precision {p} recall {r} (TP {s['tp']}, FP {s['fp']}, FN {s['fn']})"


def load_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in READ_COLUMNS)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df.sort_values("trans_date_trans_time", kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=config.DATA_DIR / config.TEST_FILE,
                        help="transactions to stream (Kaggle schema)")
    parser.add_argument("--warmup", type=Path, nargs="*", default=None,
                        help="earlier transactions to load into the feature store (default: fraudTrain.csv)")
    parser.add_argument("--start", default=None, help="first timestamp to stream; earlier rows only warm up")
    parser.add_argument("--limit", type=int, default=2000, help="number of transactions to stream (0 = all)")
    parser.add_argument("--batch-size", type=int, default=1, help="1 = one request per transaction")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--quiet", action="store_true", help="don't print individual alerts")
    parser.add_argument("--out", type=Path, default=None, help="optional CSV of every prediction")
    args = parser.parse_args()

    try:
        health = json.loads(urllib.request.urlopen(args.api.rstrip("/") + "/health", timeout=5).read())
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"API not reachable at {args.api} ({exc}). Start it first with `make serve`.")
    if not health.get("model_loaded"):
        raise SystemExit(f"API is up but has no model: {health.get('error')}")

    store = CardHistoryStore()
    warmup_files = args.warmup if args.warmup is not None else (
        [config.DATA_DIR / config.TRAIN_FILE] if (config.DATA_DIR / config.TRAIN_FILE).exists() else [])
    t0 = time.time()
    for path in warmup_files:
        store.warm_up(load_transactions(path))
    stream = load_transactions(args.file)
    if args.start:
        before = stream[stream["trans_date_trans_time"] < pd.Timestamp(args.start)]
        if len(before):
            store.warm_up(before)
        stream = stream[stream["trans_date_trans_time"] >= pd.Timestamp(args.start)]
    if args.limit:
        stream = stream.head(args.limit)
    print(f"Feature store warmed up with {len(store):,} cards in {time.time() - t0:.1f}s. "
          f"Streaming {len(stream):,} transactions to {args.api} "
          f"({'one request each' if args.batch_size == 1 else f'batches of {args.batch_size}'})...\n")

    t0 = time.time()
    stats = replay(stream.to_dict("records"), store, http_scorer(args.api), args.batch_size, not args.quiet)
    elapsed = time.time() - t0

    print(f"\nDone: {stats['scored']:,} transactions in {elapsed:.1f}s, {stats['alerts']} alerts.")
    if stats["labelled"]:
        print("Against the labels:" + _running(stats))
    if stats["latency_ms_p50"] is not None:
        print(f"Client-side latency per transaction: p50 {stats['latency_ms_p50']:.1f} ms, "
              f"p95 {stats['latency_ms_p95']:.1f} ms")
    if args.out:
        pd.DataFrame(stats["results"]).to_csv(args.out, index=False)
        print(f"Predictions written to {args.out}")


if __name__ == "__main__":
    main()
