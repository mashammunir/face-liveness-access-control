"""
Central configuration for the Face Liveness Access Control System.

All tunable paths and thresholds live here so nothing is hard-coded
inside the application logic. Values can be overridden via environment
variables (see .env.example) without touching any source code.
"""

import os
from pathlib import Path

# Load .env file if python-dotenv is available (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Project root ---
BASE_DIR = Path(__file__).resolve().parent

# --- Paths (all relative to project root, overridable via env vars) ---
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "liveness_model.pth"))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", BASE_DIR / "models" / "users.db"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", BASE_DIR / "results"))
ACCESS_LOG_PATH = Path(os.getenv("ACCESS_LOG_PATH", BASE_DIR / "models" / "access_log.db"))

# --- Camera ---
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# --- Face detection ---
# Haar Cascade is used for simplicity/reliability with zero extra installs.
HAAR_CASCADE_PATH = os.getenv("HAAR_CASCADE_PATH", "haarcascade_frontalface_default.xml")
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "80"))  # pixels, in original frame

# --- Liveness detection ---
LIVENESS_THRESHOLD = float(os.getenv("LIVENESS_THRESHOLD", "0.5"))  # prob >= this => LIVE
LIVENESS_INPUT_SIZE = int(os.getenv("LIVENESS_INPUT_SIZE", "128"))  # square crop size fed to CNN

# --- Face recognition ---
FACE_RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", "0.6"))  # cosine similarity

# --- Training defaults (used by scripts/train_liveness.py) ---
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "32"))
TRAIN_LEARNING_RATE = float(os.getenv("TRAIN_LEARNING_RATE", "1e-4"))
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "15"))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "5"))

# Ensure key directories exist
for _dir in (MODEL_PATH.parent, DATABASE_PATH.parent, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
