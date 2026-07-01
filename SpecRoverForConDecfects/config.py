from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
CONDEFECTS_ROOT = REPO_ROOT / "ConDefects"

DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "SPECROVER_CONDEFECTS_DATASET_ROOT",
        CONDEFECTS_ROOT / "datasetForExperiment",
    )
)
DEFAULT_TEST_ROOT = Path(
    os.environ.get("SPECROVER_CONDEFECTS_TEST_ROOT", CONDEFECTS_ROOT / "Test")
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("SPECROVER_CONDEFECTS_OUTPUT_ROOT", PACKAGE_ROOT / "output")
)
DEFAULT_WORK_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "batch_repair_datasetForExperiment_work"
DEFAULT_RESULTS_DIR = DEFAULT_OUTPUT_ROOT / "eval_specrover_validation_feedback_module_Ochiai"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_ROOT / "batch_repair_datasetForExperiment_summary.json"

DEFAULT_MODEL_NAME = os.environ.get(
    "SPECROVER_CONDEFECTS_MODEL",
    "gpt-4o",
)
DEFAULT_TEMPERATURE = float(os.environ.get("SPECROVER_CONDEFECTS_TEMPERATURE", "0.0"))
DEFAULT_REPAIR_RUNS = int(os.environ.get("SPECROVER_CONDEFECTS_REPAIR_RUNS", "1"))
DEFAULT_PATCH_RETRIES = int(os.environ.get("SPECROVER_CONDEFECTS_PATCH_RETRIES", "1"))

