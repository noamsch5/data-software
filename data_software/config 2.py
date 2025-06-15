from typing import Dict
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

# Platform payout rates
PLATFORM_PAYOUTS: Dict[str, float] = {
    'spotify': float(os.getenv('PLATFORM_PAYOUT_SPOTIFY', '0.0035')),
    'apple': float(os.getenv('PLATFORM_PAYOUT_APPLE', '0.005')),
    'youtube': float(os.getenv('PLATFORM_PAYOUT_YOUTUBE', '0.001')),
    'beatport': float(os.getenv('PLATFORM_PAYOUT_BEATPORT', '0.75'))
}

# נתיבים
BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / 'data_raw'
OUTPUT_DIR = BASE_DIR / 'output'
MODELS_DIR = BASE_DIR / 'models'

# מסד נתונים
DATABASE_URL = f"sqlite:///{BASE_DIR}/mvp.db"

# פלטפורמות
PLATFORMS = ['Spotify', 'Sales', 'Overall']

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
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10.0,
    'holidays_prior_scale': 10.0,
    'seasonality_mode': 'multiplicative'
}

# הגדרות XGBoost
XGBOOST_SETTINGS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'reg:squarederror'
}

# הגדרות LSTM
LSTM_SETTINGS = {
    'units': 50,
    'dropout': 0.2,
    'epochs': 100,
    'batch_size': 32
}

# יצירת תיקיות אם לא קיימות
for dir_path in [RAW_DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Database settings
DB_PATH = 'mvp.db'
SAMPLE_DATA_DIR = 'sample_data'

# Prophet settings
FORECAST_DAYS = 90
PROPHET_PARAMS = {
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10.0,
    'seasonality_mode': 'multiplicative'
} 