import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database - with fallbacks
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ml_cost_governance")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Build connection URL
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Paths (using Path objects)
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
MODELS_SAVED = Path("models/saved")

# Model settings
TRAIN_SPLIT = 0.70
PREDICTION_POINT = 0.20    # predict at 20% project completion
MAPE_TARGET = 15.0          # target MAPE threshold
N_ESTIMATORS = 100          # Random Forest trees
RANDOM_STATE = 42

# Governance thresholds
TIER_GREEN = 5.0
TIER_AMBER = 10.0
TIER_RED = 20.0
# TIER_CRITICAL is anything above RED