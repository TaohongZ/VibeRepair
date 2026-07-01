from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
CONDEFECTS_ROOT = REPO_ROOT / "ConDefects"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

DEFAULT_MODEL_NAME = "gpt-4o"
DEFAULT_AGENT_MODEL_NAME = DEFAULT_MODEL_NAME
DEFAULT_PATCH_MODEL_NAME = DEFAULT_MODEL_NAME
EMBEDDING_MODEL_NAME = "text-embedding-3-large"

DEFAULT_DATASET_ROOT = Path(
    os.environ.get("REINFIX_CONDEFECTS_DATASET_ROOT", CONDEFECTS_ROOT / "datasetForExperiment")
)
DEFAULT_TEST_ROOT = Path(os.environ.get("REINFIX_CONDEFECTS_TEST_ROOT", CONDEFECTS_ROOT / "Test"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("REINFIX_CONDEFECTS_OUTPUT_ROOT", PACKAGE_ROOT / "output"))

RETRIEVAL_CSV_PATH = Path(
    os.environ.get(
        "REINFIX_CONDEFECTS_RETRIEVAL_CSV",
        PACKAGE_ROOT / "retrieval_base" / "embedded_cause_97k.csv",
    )
)
RETRIEVAL_PARQUET_PATH = Path(
    os.environ.get(
        "REINFIX_CONDEFECTS_RETRIEVAL_PARQUET",
        PACKAGE_ROOT / "retrieval_base" / "embedded_cause_97k.parquet",
    )
)
RETRIEVAL_PICKLE_PATH = Path(
    os.environ.get(
        "REINFIX_CONDEFECTS_RETRIEVAL_PICKLE",
        PACKAGE_ROOT / "retrieval_base" / "embedded_cause_97k.pkl",
    )
)

RAG_SEARCH_URL = os.environ.get("REINFIX_CONDEFECTS_SEARCH_URL", "http://localhost:5000/search")

