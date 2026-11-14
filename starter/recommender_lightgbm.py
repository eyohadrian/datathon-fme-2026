"""LightGBM LambdaRank recommender for the RawTree FME Datathon.

Place this file at starter/recommender.py, next to starter/lightgbm_ranker.py.

Before starting, save the fitted ShopRanker from your notebook:
    ranker.save('../data/lgbm_ranker.joblib')

Run from the repository root:
    uv run --no-project python starter/recommender.py --dry-run
    uv run --no-project python starter/recommender.py

The fitted ShopRanker artifact includes its catalogue, popularity ranks and
Item2Vec vectors. It must be the same trained object evaluated in the notebook.
Only load joblib files produced by you or another trusted source.

This script pulls new requests, predicts exactly ten eligible products and
posts answers in batches. No retraining or network lookup is done per request.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# Import ShopRanker from the file in the same `starter/` directory.
from lightgbm_ranker import ShopRanker

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
MODEL_VERSION = "lightgbm-lambdarank-v1"
POLL_SECONDS = 0.30
STOCK_REFRESH_SECONDS = 1.0
REQUEST_LIMIT = 50
MAX_REQUEST_AGE_S = 3.0  # Skip requests that have already missed the deadline.


def load_env() -> None:
    """Small .env reader; already-exported environment variables take priority."""
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()
MODEL_PATH = Path(os.environ.get("LIGHTGBM_MODEL_PATH", str(ROOT / "data" / "lgbm_ranker.joblib")))
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ.get("TEAM_ID", "")
API_KEY = os.environ.get("RAWTREE_API_KEY", "")

http = requests.Session()
if API_KEY:
    http.headers["Authorization"] = f"Bearer {API_KEY}"

Record = dict[str, Any]

# RawTree may return recent_views as a JSON array and context as a JSON object.
NEW_REQUESTS = f"""
SELECT
    request_id::String AS request_id,
    ts,
    recent_views,
    context,
    is_new_user
FROM shop.requests
ORDER BY ts DESC
LIMIT {REQUEST_LIMIT}
"""

# Latest stock event for every product that has ever had a stock change.
OUT_OF_STOCK = """
SELECT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'
"""


def query(sql: str) -> list[Record]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=10)
    if not r.ok:
        print(f"RawTree query error ({r.status_code}): {r.text[:900]}", flush=True)
    r.raise_for_status()
    return r.json()["data"]


def push(answers: list[Record]) -> None:
    if not answers:
        return
    r = http.post(
        f"{API}/v1/tables/submissions",
        params={"database": TEAM},
        json=answers,
        timeout=10,
    )
    if not r.ok:
        print(f"Submission error ({r.status_code}): {r.text[:900]}", flush=True)
    r.raise_for_status()
    print(f"answered {len(answers)} requests", flush=True)


def load_ranker() -> ShopRanker:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing trained LightGBM model: {MODEL_PATH}\n"
            "From the notebook, run ranker.save('../data/lgbm_ranker.joblib') "
            "and place the output under data/."
        )
    ranker = ShopRanker.load(MODEL_PATH)
    if ranker.estimator is None:
        raise ValueError("The saved ShopRanker has not been fitted")

    # Repair a model saved with an older version of lightgbm_ranker.py.
    if not hasattr(ranker, "active_ids"):
        catalog = ranker.catalog
        if "is_active" in catalog.columns:
            active = catalog["is_active"].astype(str).str.lower().isin(("true", "1"))
            ranker.active_ids = set(catalog.loc[active, "product_id"].astype(str))
        else:
            ranker.active_ids = set(catalog["product_id"].astype(str))

    print(
        f"Loaded LightGBM: {len(ranker.active_ids)} active catalog products, "
        f"{len(ranker.best)} popularity-ranked products, "
        f"{len(ranker.FEATURE_NAMES)} features",
        flush=True,
    )
    return ranker


def out_of_stock() -> set[str]:
    return {str(r["product_id"]) for r in query(OUT_OF_STOCK)}


def pull(seen: set[str]) -> list[Record]:
    """Fetch recent requests; mark them seen only after a successful POST."""
    rows = query(NEW_REQUESTS)
    # Oldest first, while still retrieving only the newest 50 requests.
    return [row for row in reversed(rows) if row["request_id"] not in seen]


def is_request_too_old(req: Record) -> bool:
    """Do not submit an answer for a request already past the 3s deadline.

    The organizers use their clock, so this relies on a reasonably synced local
    system clock. Requests with malformed timestamps are passed through.
    """
    value = req.get("ts")
    if not isinstance(value, str):
        return False
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds() > MAX_REQUEST_AGE_S
    except ValueError:
        return False


def parse_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in ("1", "true", "yes")


def fallback_items(ranker: ShopRanker, available: set[str]) -> list[str]:
    """Use the ranker's training-time popularity list, then active catalogue."""
    result: list[str] = []
    used: set[str] = set()
    for pid in list(ranker.best) + sorted(available):
        if pid in available and pid not in used:
            result.append(pid)
            used.add(pid)
            if len(result) == 10:
                break
    if len(result) != 10:
        raise RuntimeError(f"Cannot find ten active, in-stock items: {len(result)}")
    return result


def predict(
    incoming: list[Record],
    ranker: ShopRanker,
    unavailable: set[str],
    available: set[str],
    fallback: list[str],
) -> tuple[list[Record], set[str]]:
    """Return valid submissions and locally-skipped request IDs."""
    answers: list[Record] = []
    expired: set[str] = set()
    for req in incoming:
        rid = str(req["request_id"])
        if is_request_too_old(req):
            expired.add(rid)
            continue
        version = MODEL_VERSION
        try:
            items = ranker.recommend(
                recent_views=req.get("recent_views") or [],
                context=req.get("context") or {},
                is_new_user=parse_bool(req.get("is_new_user")),
                unavailable=unavailable,
            )
            # A stock snapshot from RawTree may be fresher than the model's
            # catalog, so enforce eligibility again before submitting.
            if len(items) != 10 or len(set(items)) != 10 or not set(items) <= available:
                raise ValueError("invalid or unavailable items returned by ranker")
        except Exception as e:
            # Continue serving safely if one prediction fails.
            print(f"LightGBM failed for {rid}: {e}; using local fallback", flush=True)
            items = fallback
            version = "lightgbm-local-fallback"

        answers.append({
            "request_id": rid,
            "items": items,
            "model_version": version,
        })
    return answers, expired


def main(dry_run: bool = False) -> None:
    if not dry_run and (not TEAM or not API_KEY):
        raise RuntimeError("Set TEAM_ID and RAWTREE_API_KEY in .env or the environment")

    ranker = load_ranker()
    initial_active = set(ranker.active_ids)
    unavailable: set[str] = set()
    try:
        unavailable = out_of_stock() if API_KEY else set()
    except Exception as e:
        # Before the event, catalog_updates might not be available yet.
        print(f"Initial stock query unavailable: {e}", flush=True)
    available = initial_active - unavailable
    fallback = fallback_items(ranker, available)

    if dry_run:
        demo = {
            "request_id": "demo-not-submitted",
            "recent_views": [{"product_id": pid} for pid in fallback[:3]],
            "context": {"device": "ios", "region": "Catalunya", "page": "product"},
            "is_new_user": False,
        }
        answers, _ = predict([demo], ranker, unavailable, available, fallback)
        print(f"Dry-run output: {answers[0]}", flush=True)
        print("Nothing was submitted to RawTree.", flush=True)
        return

    print(f"Starting live polling for {TEAM}; model version {MODEL_VERSION}", flush=True)
    seen: set[str] = set()
    stock_checked = time.monotonic()

    while True:
        try:
            if time.monotonic() - stock_checked >= STOCK_REFRESH_SECONDS:
                stock_checked = time.monotonic()
                try:
                    unavailable = out_of_stock()
                    available = initial_active - unavailable
                    fallback = fallback_items(ranker, available)
                except Exception as e:
                    print(f"Stock check failed, keeping previous snapshot: {e}", flush=True)

            incoming = pull(seen)
            if incoming:
                answers, expired = predict(incoming, ranker, unavailable, available, fallback)
                if expired:
                    seen.update(expired)
                    print(f"Skipped {len(expired)} expired requests", flush=True)
                if answers:
                    push(answers)
                    seen.update(ans["request_id"] for ans in answers)

        except Exception as e:
            print(f"Loop error: {e}", flush=True)
            time.sleep(1.0)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
