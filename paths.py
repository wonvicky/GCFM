"""Shared project paths for GCFM (new_code3)."""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(ROOT_DIR, "result")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Override on server via env: export ICDM_DATA_ROOT=/root/ICDM
DATA_ROOT = os.environ.get("ICDM_DATA_ROOT", "/root/ICDM")
DIFFSTG_DIR = os.environ.get("DIFFSTG_DIR", os.path.join(DATA_ROOT, "DiffSTG-main"))
