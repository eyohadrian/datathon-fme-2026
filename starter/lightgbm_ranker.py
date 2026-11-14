"""LightGBM LambdaRank policy for the RawTree FME Datathon.

Notebook usage:
    from lightgbm_ranker import ShopRanker, sample_history_csv
    history = sample_history_csv('../data/history_sessions.csv', fraction=0.10)
    history = history[history.ts < pd.Timestamp(CUT, tz='UTC')]
    ranker = ShopRanker(catalogos, best, item2vec=model)
    ranker.fit(history)
    ranker.save('../data/lgbm_ranker.joblib')
    POLICIES['lgbm'] = ranker.policy
    evaluate(ranker.policy, test)

Note: history must be strictly earlier than the evaluation window, and the
Word2Vec model and popularity ranking must also have been built using data
available before that window. Historical carousel impressions have selection
and position bias. Offline rank metrics are not the hidden conversion score.
"""
from __future__ import annotations

import ast
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


def _parse(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return default
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return default
    if isinstance(value, str):
        if not value.strip():
            return default
        for parse in (json.loads, ast.literal_eval):
            try:
                x = parse(value)
                if isinstance(x, type(default)):
                    return x
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                pass
    return default


def _str(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    s = str(x)
    return "" if s in ("NC", "NP", "NA", "nan", "None", "<NA>") else s


def _ts_utc(values: pd.Series) -> pd.Series:
    """Handle ISO timestamps and Unix seconds / milliseconds in user CSVs."""
    non_null = values.dropna()
    if not non_null.empty and (
        pd.api.types.is_numeric_dtype(values)
        or str(non_null.iloc[0]).replace('.', '', 1).isdigit()
    ):
        n = pd.to_numeric(values, errors="coerce")
        unit = "ms" if n.dropna().abs().median() > 1e11 else "s"
        return pd.to_datetime(n, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def sample_history_csv(path: str | Path, fraction: float = 0.10, seed: int = 42) -> pd.DataFrame:
    """Read a reproducible fraction of a large CSV, keeping only useful columns.

    Reads the entire CSV sequentially but retains only ~fraction of its rows.
    The caller still needs to apply a strict time cutoff before model training.
    """
    cols = ["request_id", "ts", "recent_views", "shown", "clicked", "purchased", "context"]
    rng = np.random.default_rng(seed)
    kept = []
    for chunk in pd.read_csv(path, usecols=lambda col: col in cols, chunksize=40000):
        sample = chunk.iloc[np.flatnonzero(rng.random(len(chunk)) < fraction)].copy()
        if not sample.empty:
            kept.append(sample)
    if not kept:
        raise ValueError("No rows sampled: increase fraction")
    out = pd.concat(kept, ignore_index=True)
    out["ts"] = _ts_utc(out["ts"])
    return out.dropna(subset=["ts", "request_id"])


class ShopRanker:
    """Re-ranks item2vec + popularity candidate products using LambdaRank."""

    FEATURE_NAMES = [
        "pop_rank_score", "log_price", "n_recent", "recently_viewed",
        "matches_category", "matches_subcategory", "matches_brand",
        "price_distance", "embedding_max_cos", "embedding_mean_cos",
        "page_home", "page_product", "page_category",
        "device_web", "device_ios", "device_android", "is_new_user",
    ]

    def __init__(self, catalog: pd.DataFrame, best: list[str], item2vec=None):
        self.catalog = catalog.copy().drop_duplicates("product_id")
        self.catalog["product_id"] = self.catalog["product_id"].astype(str)
        self.catalog_by_id = self.catalog.set_index("product_id")
        # The initial catalog can contain inactive products. Only offer active IDs.
        if "is_active" in self.catalog.columns:
            active_mask = self.catalog["is_active"].astype(str).str.lower().isin(("true", "1"))
            self.active_ids = set(self.catalog.loc[active_mask, "product_id"])
        else:
            self.active_ids = set(self.catalog["product_id"])
        self.best = list(dict.fromkeys(str(p) for p in best))
        self.pop_rank = {pid: i for i, pid in enumerate(self.best)}
        # Gensim is optional: a ranker without embeddings remains runnable,
        # but validate with the same feature configuration used at serving.
        self.wv = getattr(item2vec, "wv", item2vec)
        self.estimator = None
        self._neighbor_cache: dict[str, list[str]] = {}
        self._id_meta = {}
        price_series = pd.to_numeric(self.catalog.get("price_eur"), errors="coerce")
        self._median_price = float(price_series.dropna().median()) if price_series.notna().any() else 0.0
        for _, r in self.catalog.iterrows():
            self._id_meta[str(r["product_id"])] = (
                _str(r.get("category")),
                _str(r.get("subcategory")),
                _str(r.get("brand")),
                self._number(r.get("price_eur"), self._median_price),
            )

    @staticmethod
    def _number(v: Any, default: float) -> float:
        try:
            x = float(v)
            return x if np.isfinite(x) else default
        except (ValueError, TypeError):
            return default

    def _views(self, raw: Any) -> list[str]:
        values = _parse(raw, [])
        return [
            str(v["product_id"]) if isinstance(v, dict) else str(v)
            for v in values
            if (isinstance(v, dict) and v.get("product_id")) or isinstance(v, str)
        ]

    def _embed_sims(self, pid: str, views: list[str]) -> tuple[float, float]:
        if self.wv is None or pid not in self.wv:
            return 0.0, 0.0
        refs = [v for v in views[-5:] if v in self.wv]
        if not refs:
            return 0.0, 0.0
        # Gensim similarity is normalized cosine similarity. Clip negative
        # values to zero: this is a proximity signal, not a label.
        vals = [max(0.0, float(self.wv.similarity(pid, r))) for r in refs]
        return max(vals), float(np.mean(vals))

    def features(self, candidates: list[str], recent_views: Any,
                 context: Any = None, is_new_user: bool = False) -> pd.DataFrame:
        views = self._views(recent_views)
        view_set = set(views)
        context = _parse(context, {})
        device = _str(context.get("device", ""))
        page = _str(context.get("page", ""))
        recent_meta = [self._id_meta[v] for v in views[-5:] if v in self._id_meta]
        prices = [m[3] for m in recent_meta]
        reference_price = float(np.mean(prices)) if prices else self._median_price
        categories = {m[0] for m in recent_meta if m[0]}
        subcategories = {m[1] for m in recent_meta if m[1]}
        brands = {m[2] for m in recent_meta if m[2]}
        out = []
        for pid in candidates:
            cat, sub, brand, price = self._id_meta.get(pid, ("", "", "", self._median_price))
            rank = self.pop_rank.get(pid)
            mx, mean = self._embed_sims(pid, views)
            out.append([
                1.0 / np.log2(rank + 2) if rank is not None else 0.0,
                np.log1p(max(0.0, price)),
                min(10, len(views)),
                float(pid in view_set),
                float(bool(cat) and cat in categories),
                float(bool(sub) and sub in subcategories),
                float(bool(brand) and brand in brands),
                abs(np.log1p(max(0.0, price)) - np.log1p(max(0.0, reference_price))),
                mx, mean,
                float(page == "home"), float(page == "product"), float(page == "category"),
                float(device == "web"), float(device == "ios"), float(device == "android"),
                float(bool(is_new_user)),
            ])
        return pd.DataFrame(out, columns=self.FEATURE_NAMES, dtype="float32")

    def _frame_from_impressions(self, df: pd.DataFrame, positive_only: bool = True):
        matrices, labels, groups = [], [], []
        for row in df.itertuples(index=False):
            shown = [str(p) for p in _parse(getattr(row, "shown", None), [])]
            if len(shown) != 10 or len(set(shown)) != 10:
                continue
            clicked = _str(getattr(row, "clicked", None))
            purchased = _str(getattr(row, "purchased", None))
            y = np.asarray([
                2 if p == purchased else 1 if p == clicked else 0
                for p in shown
            ], dtype="int8")
            # No training gradient for slates that have no clicks/purchases.
            if positive_only and y.max() == 0:
                continue
            X = self.features(shown, getattr(row, "recent_views", None),
                              getattr(row, "context", None),
                              getattr(row, "is_new_user", False))
            matrices.append(X.to_numpy(dtype="float32", copy=False))
            labels.append(y)
            groups.append(len(shown))
        if not matrices:
            raise ValueError("No valid impression groups; inspect shown/clicked/purchased")
        return np.concatenate(matrices), np.concatenate(labels), groups

    def fit(self, history: pd.DataFrame, validation_fraction: float = 0.2):
        """Fit on historical impression groups only, split chronologically."""
        if "ts" not in history.columns:
            raise ValueError("history must contain ts for a temporal split")
        history = history.copy()
        history["ts"] = _ts_utc(history["ts"])
        history = history.dropna(subset=["ts"]).sort_values(["ts", "request_id"])
        split = int(len(history) * (1 - validation_fraction))
        if split < 1 or split >= len(history):
            raise ValueError("Need a larger training sample")
        train = history.iloc[:split]
        val = history.iloc[split:]
        print(f"Train impressions: {len(train):,}; temporal validation: {len(val):,}")
        Xtr, ytr, gtr = self._frame_from_impressions(train)
        Xval, yval, gval = self._frame_from_impressions(val)
        print(f"Training positive slates: {len(gtr):,}; validation positive slates: {len(gval):,}")
        self.estimator = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg", label_gain=[0, 1, 8],
            n_estimators=350, learning_rate=0.04, num_leaves=15,
            min_child_samples=60, colsample_bytree=0.9, subsample=0.9,
            reg_lambda=5.0, random_state=42, n_jobs=4, verbosity=-1,
        )
        self.estimator.fit(
            Xtr, ytr, group=gtr,
            eval_set=[(Xval, yval)], eval_group=[gval],
            eval_at=(1, 10),
            callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)],
        )
        print("Best iteration:", self.estimator.best_iteration_)
        print("Validation NDCG:", self.estimator.best_score_.get("valid_0"))
        importance = pd.Series(self.estimator.feature_importances_, index=self.FEATURE_NAMES)
        print("Feature importance:\n", importance.sort_values(ascending=False).head(10))
        return self

    def _neighbors(self, pid: str, topn: int = 80) -> list[str]:
        if self.wv is None or pid not in self.wv:
            return []
        if pid not in self._neighbor_cache:
            self._neighbor_cache[pid] = [p for p, _ in self.wv.most_similar(pid, topn=topn)]
        return self._neighbor_cache[pid]

    def candidates(self, recent_views: Any) -> list[str]:
        result = list(self.best[:200])
        for pid in self._views(recent_views)[-3:]:
            result.append(pid)
            result.extend(self._neighbors(pid))
        return list(dict.fromkeys(result))

    def recommend(self, recent_views: Any, context: Any = None,
                  is_new_user: bool = False, unavailable: set[str] | frozenset[str] = frozenset()) -> list[str]:
        if self.estimator is None:
            raise RuntimeError("Fit or load the ranker first")
        candidates = [
            p for p in self.candidates(recent_views)
            if p in self.active_ids and p not in unavailable
        ]
        # Recover from a stale popularity list or a large stock-out event.
        if len(candidates) < 10:
            candidates.extend(
                p for p in sorted(self.active_ids)
                if p not in unavailable and p not in candidates
            )
        if len(candidates) < 10:
            raise ValueError("Fewer than 10 available candidates")
        X = self.features(candidates, recent_views, context, is_new_user)
        scores = self.estimator.predict(X)
        order = np.argsort(-scores, kind="stable")
        return [candidates[int(i)] for i in order[:10]]

    def policy(self, row) -> list[str]:
        # Works with rows created by df.itertuples(), like evaluate().
        return self.recommend(
            getattr(row, "recent_views", None),
            getattr(row, "context", None),
            getattr(row, "is_new_user", False),
        )

    def save(self, path: str | Path) -> None:
        # Joblib uses pickle; only load files you created or trust.
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> "ShopRanker":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ShopRanker; got {type(obj)}")
        return obj
