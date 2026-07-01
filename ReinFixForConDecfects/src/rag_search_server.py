from __future__ import annotations

import argparse
import ast
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from openai import OpenAI

from .config import (
    EMBEDDING_MODEL_NAME,
    OPENAI_API_KEY,
    RETRIEVAL_CSV_PATH,
    RETRIEVAL_PARQUET_PATH,
    RETRIEVAL_PICKLE_PATH,
)


if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

app = Flask(__name__)
client = OpenAI()


def _safe_eval_embedding(value):
    try:
        parsed = ast.literal_eval(value)
        if parsed is None:
            return None
        embedding = np.asarray(parsed, dtype=np.float32)
        if embedding.ndim != 1 or embedding.size == 0:
            return None
        return embedding
    except Exception:
        return None


@lru_cache(maxsize=None)
def load_data(csv_path: str) -> pd.DataFrame:
    csv_path_obj = Path(csv_path)
    parquet_path = RETRIEVAL_PARQUET_PATH
    pickle_path = RETRIEVAL_PICKLE_PATH

    if parquet_path.exists():
        print(f"Loading data from parquet cache: {parquet_path}", flush=True)
        df = pd.read_parquet(parquet_path)
        df["ada_embedding"] = df["ada_embedding"].apply(lambda value: np.asarray(value, dtype=np.float32))
        print("Loaded parquet cache successfully.", flush=True)
        return df

    if pickle_path.exists():
        print(f"Loading data from pickle cache: {pickle_path}", flush=True)
        df = pd.read_pickle(pickle_path)
        df["ada_embedding"] = df["ada_embedding"].apply(lambda value: np.asarray(value, dtype=np.float32))
        print("Loaded pickle cache successfully.", flush=True)
        return df

    print(f"Parquet cache not found. Loading CSV: {csv_path_obj}", flush=True)
    df = pd.read_csv(csv_path_obj)
    print("CSV loaded. Parsing embeddings...", flush=True)

    df["ada_embedding"] = df["ada_embedding"].apply(_safe_eval_embedding)
    df = df.dropna(subset=["ada_embedding"])

    keep_cols = []
    for column in ["BuggyCode", "FixedCode", "RootCause", "ada_embedding"]:
        if column in df.columns:
            keep_cols.append(column)
    df = df[keep_cols]

    cache_df = df.copy()
    cache_df["ada_embedding"] = cache_df["ada_embedding"].apply(lambda value: value.tolist())
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Saving parquet cache to: {parquet_path}", flush=True)
        cache_df.to_parquet(parquet_path, index=False)
        print("Parquet cache saved successfully.", flush=True)
    except Exception as exc:
        print(f"Could not save parquet cache ({exc}). Saving pickle cache to: {pickle_path}", flush=True)
        cache_df.to_pickle(pickle_path)
        print("Pickle cache saved successfully.", flush=True)

    return df


def get_embedding(text: str, model: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
    return np.asarray(embedding, dtype=np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def search_reviews(df: pd.DataFrame, input_pattern: str, n: int, threshold: float) -> pd.DataFrame:
    input_embedding = get_embedding(input_pattern, model=EMBEDDING_MODEL_NAME)
    scored = df.copy()
    scored["similarity"] = scored["ada_embedding"].apply(
        lambda embedding: cosine_similarity(embedding, input_embedding)
    )
    filtered_df = scored[scored["similarity"] > threshold]

    if filtered_df.empty:
        return pd.DataFrame()

    return filtered_df.sort_values("similarity", ascending=False).head(n)


def search_hybrid(
    df: pd.DataFrame,
    input_pattern: str,
    n: int,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    input_embedding = get_embedding(input_pattern, model=EMBEDDING_MODEL_NAME)

    scored = df.copy()
    scored["similarity"] = scored["ada_embedding"].apply(
        lambda embedding: cosine_similarity(embedding, input_embedding)
    )
    scored = scored.sort_values("similarity", ascending=False)

    meta = {
        "mode": "topk_only" if threshold is None else "threshold_then_topk",
        "threshold": threshold,
        "filled_from_topk": False,
        "threshold_count": None,
    }

    if threshold is None:
        return scored.head(n), meta

    threshold_part = scored[scored["similarity"] > float(threshold)]
    meta["threshold_count"] = int(len(threshold_part))
    results = threshold_part.head(n)

    if len(results) < n:
        meta["filled_from_topk"] = True
        need = n - len(results)
        filler = scored.loc[~scored.index.isin(results.index)].head(need)
        results = pd.concat([results, filler], axis=0)

    return results, meta


print("Data loading...", flush=True)
df = load_data(str(RETRIEVAL_CSV_PATH))
print("Data loaded successfully.", flush=True)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "rows": int(len(df)),
            "embedding_model": EMBEDDING_MODEL_NAME,
        }
    )


@app.route("/search", methods=["POST"])
def search():
    data = request.json or {}
    query = data.get("query", "")

    try:
        n = int(data.get("n", 5))
    except Exception:
        n = 5

    raw_threshold = data.get("threshold", None)
    threshold = None
    if raw_threshold not in (None, "", "null"):
        try:
            threshold = float(raw_threshold)
        except Exception:
            threshold = None

    if not query:
        return jsonify({"error": "No query provided"}), 400

    results, meta = search_hybrid(df, query, n=n, threshold=threshold)

    if results.empty:
        return jsonify([{"message": "No examples found"}])

    response = []
    for _, row in results.iterrows():
        response.append(
            {
                "similarity": float(row["similarity"]),
                "buggy_code": row.get("BuggyCode", ""),
                "fixed_code": row.get("FixedCode", ""),
                "root_cause": row.get("RootCause", ""),
                "meta": meta,
            }
        )

    return jsonify(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ReInFix example-patch RAG server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)

