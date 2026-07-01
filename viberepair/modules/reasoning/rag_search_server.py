from __future__ import annotations

import argparse
import ast
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised in minimal environments
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised in minimal environments
    pd = None

try:
    from flask import Flask, jsonify, request
except ImportError:  # pragma: no cover - exercised in minimal environments
    Flask = None
    jsonify = None
    request = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised in minimal environments
    OpenAI = None


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RETRIEVAL_BASE = Path(
    os.environ.get(
        "VIBEREPAIR_REASONING_RETRIEVAL_BASE",
        REPO_ROOT / "ReinFixForConDecfects" / "retrieval_base",
    )
)
RETRIEVAL_CSV_PATH = Path(
    os.environ.get(
        "VIBEREPAIR_REASONING_RETRIEVAL_CSV",
        DEFAULT_RETRIEVAL_BASE / "embedded_cause_97k.csv",
    )
)
RETRIEVAL_PARQUET_PATH = Path(
    os.environ.get(
        "VIBEREPAIR_REASONING_RETRIEVAL_PARQUET",
        DEFAULT_RETRIEVAL_BASE / "embedded_cause_97k.parquet",
    )
)
RETRIEVAL_PICKLE_PATH = Path(
    os.environ.get(
        "VIBEREPAIR_REASONING_RETRIEVAL_PICKLE",
        DEFAULT_RETRIEVAL_BASE / "embedded_cause_97k.pkl",
    )
)
EMBEDDING_MODEL_NAME = os.environ.get(
    "VIBEREPAIR_REASONING_EMBEDDING_MODEL",
    "text-embedding-3-large",
)

_OPENAI_CLIENT: Any | None = None


def _require_rag_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if pd is None:
        missing.append("pandas")
    if Flask is None:
        missing.append("flask")
    if OpenAI is None:
        missing.append("openai")
    if missing:
        raise ImportError(
            "VibeRepair reasoning RAG server requires optional dependencies: "
            + ", ".join(missing)
            + ". Install them before starting the server."
        )


def _get_openai_client() -> Any:
    _require_rag_dependencies()
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI()
    return _OPENAI_CLIENT


def _safe_eval_embedding(value: Any) -> np.ndarray | None:
    _require_rag_dependencies()
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        if parsed is None:
            return None
        embedding = np.asarray(parsed, dtype=np.float32)
        if embedding.ndim != 1 or embedding.size == 0:
            return None
        return embedding
    except Exception:
        return None


@lru_cache(maxsize=8)
def load_data(
    csv_path: str = str(RETRIEVAL_CSV_PATH),
    parquet_path: str = str(RETRIEVAL_PARQUET_PATH),
    pickle_path: str = str(RETRIEVAL_PICKLE_PATH),
) -> pd.DataFrame:
    _require_rag_dependencies()
    csv_path_obj = Path(csv_path)
    parquet_path_obj = Path(parquet_path)
    pickle_path_obj = Path(pickle_path)

    if parquet_path_obj.exists():
        print(f"Loading data from parquet cache: {parquet_path_obj}", flush=True)
        df = pd.read_parquet(parquet_path_obj)
        df["ada_embedding"] = df["ada_embedding"].apply(
            lambda value: np.asarray(value, dtype=np.float32)
        )
        print("Loaded parquet cache successfully.", flush=True)
        return df

    if pickle_path_obj.exists():
        print(f"Loading data from pickle cache: {pickle_path_obj}", flush=True)
        df = pd.read_pickle(pickle_path_obj)
        df["ada_embedding"] = df["ada_embedding"].apply(
            lambda value: np.asarray(value, dtype=np.float32)
        )
        print("Loaded pickle cache successfully.", flush=True)
        return df

    if not csv_path_obj.exists():
        raise FileNotFoundError(f"Retrieval CSV does not exist: {csv_path_obj}")

    print(f"Parquet/pickle cache not found. Loading CSV: {csv_path_obj}", flush=True)
    df = pd.read_csv(csv_path_obj)
    print("CSV loaded. Parsing embeddings...", flush=True)

    df["ada_embedding"] = df["ada_embedding"].apply(_safe_eval_embedding)
    df = df.dropna(subset=["ada_embedding"])

    keep_cols = [
        column
        for column in ["BuggyCode", "FixedCode", "RootCause", "ada_embedding"]
        if column in df.columns
    ]
    df = df[keep_cols]

    cache_df = df.copy()
    cache_df["ada_embedding"] = cache_df["ada_embedding"].apply(lambda value: value.tolist())
    parquet_path_obj.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Saving parquet cache to: {parquet_path_obj}", flush=True)
        cache_df.to_parquet(parquet_path_obj, index=False)
        print("Parquet cache saved successfully.", flush=True)
    except Exception as exc:
        print(
            f"Could not save parquet cache ({exc}). Saving pickle cache to: {pickle_path_obj}",
            flush=True,
        )
        cache_df.to_pickle(pickle_path_obj)
        print("Pickle cache saved successfully.", flush=True)

    return df


def get_embedding(text: str, model: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    client = _get_openai_client()
    embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
    return np.asarray(embedding, dtype=np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def search_hybrid(
    df: pd.DataFrame,
    input_pattern: str,
    n: int,
    threshold: float | None = None,
    embedding_model: str = EMBEDDING_MODEL_NAME,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    input_embedding = get_embedding(input_pattern, model=embedding_model)

    scored = df.copy()
    scored["similarity"] = scored["ada_embedding"].apply(
        lambda embedding: cosine_similarity(embedding, input_embedding)
    )
    scored = scored.sort_values("similarity", ascending=False)

    meta: dict[str, Any] = {
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


def create_app(
    csv_path: str = str(RETRIEVAL_CSV_PATH),
    parquet_path: str = str(RETRIEVAL_PARQUET_PATH),
    pickle_path: str = str(RETRIEVAL_PICKLE_PATH),
    embedding_model: str = EMBEDDING_MODEL_NAME,
) -> Flask:
    _require_rag_dependencies()
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "ok",
                "embedding_model": embedding_model,
                "retrieval_csv": csv_path,
                "retrieval_parquet": parquet_path,
                "retrieval_pickle": pickle_path,
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

        df = load_data(csv_path, parquet_path, pickle_path)
        results, meta = search_hybrid(
            df,
            query,
            n=n,
            threshold=threshold,
            embedding_model=embedding_model,
        )

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

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VibeRepair reasoning RAG server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--csv-path", default=str(RETRIEVAL_CSV_PATH))
    parser.add_argument("--parquet-path", default=str(RETRIEVAL_PARQUET_PATH))
    parser.add_argument("--pickle-path", default=str(RETRIEVAL_PICKLE_PATH))
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = create_app(
        csv_path=args.csv_path,
        parquet_path=args.parquet_path,
        pickle_path=args.pickle_path,
        embedding_model=args.embedding_model,
    )
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
