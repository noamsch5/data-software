from typing import Dict
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

# Platform payout rates
PLATFORM_PAYOUTS: Dict[str, float] = {
    'spotify': float(os.getenv('PLATFORM_PAYOUT_SPOTIFY', '0.0024')),
    'beatport': float(os.getenv('PLATFORM_PAYOUT_BEATPORT', '1.25'))
}

# נתיבים
BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / 'data_raw'
OUTPUT_DIR = BASE_DIR / 'output'
MODELS_DIR = BASE_DIR / 'models'

# מסד נתונים
DB_PATH = 'data/revenue.db'
DATABASE_URL = 'sqlite:///data/revenue.db'

# פלטפורמות
PLATFORMS = {
    'Spotify': 0.004,
    'Beatport': 0.5,
    'Overall': 0.0  # This will be calculated as the sum of all platforms
}

# קטגוריות עלויות
COST_CATEGORIES = [
    'Production',
    'Marketing',
    'PR',
    'Music Videos',
    'Distribution',
    'Other'
]

# סוגי קמפיינים
CAMPAIGN_TYPES = [
    'Release',
    'Tour',
    'Social Media',
    'Playlist',
    'Radio',
    'Other'
]

# סטטוסי קמפיינים
CAMPAIGN_STATUSES = [
    'Planned',
    'Active',
    'Completed',
    'Cancelled'
]

# אירועי מוזיקה חשובים
MUSIC_EVENTS = {
    'Tomorrowland': '2024-07-19',
    'ADE': '2024-10-16',
    'Coachella': '2024-04-12',
    'Ultra': '2024-03-22'
}

# הגדרות Prophet
PROPHET_SETTINGS = {
    'changepoint_prior_scale': float(os.getenv('PROPHET_CHANGEPOINT_PRIOR_SCALE', '0.05')),
    'seasonality_prior_scale': float(os.getenv('PROPHET_SEASONALITY_PRIOR_SCALE', '10.0')),
    'holidays_prior_scale': float(os.getenv('PROPHET_HOLIDAYS_PRIOR_SCALE', '10.0')),
    'seasonality_mode': os.getenv('PROPHET_SEASONALITY_MODE', 'multiplicative')
}

# הגדרות XGBoost
XGBOOST_SETTINGS = {
    'max_depth': int(os.getenv('XGB_MAX_DEPTH', '6')),
    'learning_rate': float(os.getenv('XGB_LEARNING_RATE', '0.1')),
    'n_estimators': int(os.getenv('XGB_N_ESTIMATORS', '100')),
    'objective': 'reg:squarederror'
}

# הגדרות LSTM
LSTM_SETTINGS = {
    'units': int(os.getenv('LSTM_UNITS', '50')),
    'dropout': float(os.getenv('LSTM_DROPOUT', '0.2')),
    'epochs': int(os.getenv('LSTM_EPOCHS', '100')),
    'batch_size': int(os.getenv('LSTM_BATCH_SIZE', '32'))
}

# יצירת תיקיות אם לא קיימות
for dir_path in [RAW_DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Database settings
SAMPLE_DATA_DIR = 'sample_data'

# Prophet settings
FORECAST_DAYS = 90
PROPHET_PARAMS = {
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10.0,
    'seasonality_mode': 'multiplicative'
}

# Directory settings
PROCESSED_DATA_DIR = 'processed'

# Create necessary directories
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True) 