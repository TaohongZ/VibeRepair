import os

from flask import Flask, request, jsonify
import pandas as pd
import ast
import openai
from functools import lru_cache
from openai import OpenAI
from scipy.spatial.distance import cosine
import numpy as np

#from config import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = ""

app = Flask(__name__)

client = OpenAI()


DATAFILE_PATH = "./retrieval_base/embedded_cause_97k.csv"

CACHE_PARQUET_PATH = "./retrieval_base/embedded_cause_97k.parquet"


def _safe_eval_embedding(x):
    """
    ? CSV ??? "[0.1, 0.2, ...]" ??????? numpy array?
    ?????? None?
    """
    try:
        v = ast.literal_eval(x)
        if v is None:
            return None
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            return None
        return arr
    except Exception:
        return None


@lru_cache(maxsize=None)
def load_data(csv_path: str) -> pd.DataFrame:
    """
    ??? Parquet ????;??????? CSV ????? Parquet?
    lru_cache ????:????????? load?
    """
    if os.path.exists(CACHE_PARQUET_PATH):
        print(f"Loading data from parquet cache: {CACHE_PARQUET_PATH}")
        df = pd.read_parquet(CACHE_PARQUET_PATH)
        
        df["ada_embedding"] = df["ada_embedding"].apply(lambda v: np.asarray(v, dtype=np.float32))
        print("Loaded parquet cache successfully.")
        return df

    print(f"Parquet cache not found. Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print("CSV loaded. Parsing embeddings...")

    df["ada_embedding"] = df["ada_embedding"].apply(_safe_eval_embedding)
    df = df.dropna(subset=["ada_embedding"])

    
    keep_cols = []
    for c in ["BuggyCode", "FixedCode", "RootCause", "ada_embedding"]:
        if c in df.columns:
            keep_cols.append(c)
    df = df[keep_cols]

    
    df_to_save = df.copy()
    df_to_save["ada_embedding"] = df_to_save["ada_embedding"].apply(lambda a: a.tolist())

    os.makedirs(os.path.dirname(CACHE_PARQUET_PATH), exist_ok=True)
    print(f"Saving parquet cache to: {CACHE_PARQUET_PATH}")
    df_to_save.to_parquet(CACHE_PARQUET_PATH, index=False)
    print("Parquet cache saved successfully.")

    return df


def get_embedding(text: str, model: str = "text-embedding-3-large") -> np.ndarray:
    emb = client.embeddings.create(input=[text], model=model).data[0].embedding
    return np.asarray(emb, dtype=np.float32)


def search_reviews(df: pd.DataFrame, input_pattern: str, n: int, threshold: float) -> pd.DataFrame:
    input_embedding = get_embedding(input_pattern, model="text-embedding-3-large")

    
    df = df.copy()
    df["similarity"] = df["ada_embedding"].apply(lambda x: 1.0 - cosine(x, input_embedding))

    filtered_df = df[df["similarity"] > threshold]
    if filtered_df.empty:
        return pd.DataFrame()

    return filtered_df.sort_values("similarity", ascending=False).head(n)

def search_hybrid(
    df: pd.DataFrame,
    input_pattern: str,
    n: int,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    ????:
    1) ?????? similarity ??????
    2) ???? threshold:?? similarity > threshold ???(?? n)
       - ??? n,???? top-k ??? n
    3) ???? threshold:?????? top-k(????)
    ??:(results_df, meta_info)
    """
    input_embedding = get_embedding(input_pattern, model="text-embedding-3-large")

    scored = df.copy()
    scored["similarity"] = scored["ada_embedding"].apply(lambda x: 1.0 - cosine(x, input_embedding))
    scored = scored.sort_values("similarity", ascending=False)

    meta = {
        "mode": "topk_only" if threshold is None else "threshold_then_topk",
        "threshold": threshold,
        "filled_from_topk": False,
        "threshold_count": None,
    }

    
    if threshold is None:
        return scored.head(n), meta

    
    thresh_part = scored[scored["similarity"] > float(threshold)]
    meta["threshold_count"] = int(len(thresh_part))

    results = thresh_part.head(n)

    
    if len(results) < n:
        meta["filled_from_topk"] = True
        need = n - len(results)
        
        filler = scored.loc[~scored.index.isin(results.index)].head(need)
        results = pd.concat([results, filler], axis=0)

    return results, meta


print("Data loading...")
df = load_data(DATAFILE_PATH)
print("Data loaded successfully.")


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


if __name__ == "__main__":
    
    app.run(debug=False, use_reloader=False)
