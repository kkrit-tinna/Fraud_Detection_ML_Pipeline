"""ETL entry point: extract raw CSVs -> engineer features -> load processed tables.

    python -m fraud_detection.etl [--data-dir data] [--sample-frac 0.05]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .data import extract, file_fingerprint, save_table
from .features import TARGET, build_features


def run(data_dir: Path | None = None, out_dir: Path | None = None, sample_frac: float | None = None) -> dict:
    data_dir = Path(data_dir or config.DATA_DIR)
    out_dir = Path(out_dir or config.PROCESSED_DIR)
    t0 = time.time()

    print(f"[extract] reading raw CSVs from {data_dir}")
    train_raw, test_raw = extract(data_dir)
    if sample_frac:
        # Sample whole cards (not rows) so per-card history features stay meaningful.
        cards = train_raw["cc_num"].drop_duplicates().sample(frac=sample_frac, random_state=config.RANDOM_STATE)
        train_raw = train_raw[train_raw["cc_num"].isin(cards)]
        test_raw = test_raw[test_raw["cc_num"].isin(cards)]

    print(f"[transform] engineering features for {len(train_raw):,} + {len(test_raw):,} transactions")
    train, test = build_features(train_raw, test_raw)

    print(f"[load] writing processed tables to {out_dir}")
    paths = {
        "train": str(save_table(train, out_dir / "train_features")),
        "test": str(save_table(test, out_dir / "test_features")),
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": {"train": len(train), "test": len(test)},
        "fraud_rate": {"train": float(train[TARGET].mean()), "test": float(test[TARGET].mean())},
        "raw_sha256": {
            name: file_fingerprint(data_dir / name) for name in (config.TRAIN_FILE, config.TEST_FILE)
        },
        "sample_frac": sample_frac,
        "outputs": paths,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] ETL finished in {time.time() - t0:.0f}s")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sample-frac", type=float, default=None, help="fraction of cards to keep (CI/smoke runs)")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir, args.sample_frac)


if __name__ == "__main__":
    main()
