from typing import Dict, List, Optional
import pandas as pd
import sqlite3
from prophet import Prophet
import logging
from config import DB_PATH, FORECAST_DAYS, PROPHET_PARAMS
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from prophet.serialize import model_to_json, model_from_json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import holidays
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
from config import (
    DATABASE_URL, PLATFORMS, PROPHET_SETTINGS,
    XGBOOST_SETTINGS, LSTM_SETTINGS, MUSIC_EVENTS
)
from pathlib import Path
from etl import process_distribution_statements

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    """Load and prepare data for ML model"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM monthly_revenue_total", conn)
    conn.close()
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for ML model"""
    # המרת תאריכים
    df['date'] = pd.to_datetime(df['date'])
    
    # תכונות זמן
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    
    # חגים
    us_holidays = holidays.US()
    df['is_holiday'] = df['date'].apply(lambda x: x in us_holidays).astype(int)
    
    # אירועים מיוחדים
    df['is_tomorrowland'] = ((df['month'] == 7) & (df['date'].dt.day >= 15) & (df['date'].dt.day <= 25)).astype(int)
    df['is_ade'] = ((df['month'] == 10) & (df['date'].dt.day >= 15) & (df['date'].dt.day <= 25)).astype(int)
    
    # תכונות פלטפורמה
    df['platform_encoded'] = pd.Categorical(df['platform']).codes
    
    # תכונות נוספות (רק אם revenue_usd קיימת)
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    if 'revenue_usd' in df.columns:
        df['revenue_lag_1'] = df.groupby('platform')['revenue_usd'].shift(1)
        df['revenue_lag_2'] = df.groupby('platform')['revenue_usd'].shift(2)
        df['revenue_lag_3'] = df.groupby('platform')['revenue_usd'].shift(3)
        df['revenue_ma_3'] = df.groupby('platform')['revenue_usd'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        df['revenue_ma_6'] = df.groupby('platform')['revenue_usd'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    # הדפסת העמודות אחרי יצירת הפיצ'רים
    print('COLUMNS after create_features:', df.columns.tolist())
    return df

def train_model(data, platform, period_type):
    """אימון מודל LightGBM (API גבוה)"""
    # סינון נתונים רלוונטיים בלבד
    data = data[(data['platform'].str.lower() == platform.lower()) & (data['period_type'] == period_type)].copy()
    print(f'Training model for {platform} ({period_type})')
    print('Data shape:', data.shape)
    print('Data columns:', data.columns.tolist())
    print('Last 10 rows of training data:')
    print(data.tail(10))
    
    # יצירת תכונות
    features = create_features(data)
    print('Features shape:', features.shape)
    print('Features columns:', features.columns.tolist())
    
    # הגדרת תכונות ומטרה
    feature_cols = [col for col in features.columns if col not in ['date', 'platform', 'revenue_usd', 'period_type']]
    print('Feature columns for training:', feature_cols)
    
    if len(feature_cols) == 0:
        print('No features available for training')
        return None, None, None
    
    X = features[feature_cols]
    y = features['revenue_usd']
    
    print('X shape:', X.shape)
    print('y shape:', y.shape)
    
    # חלוקה לסט אימון וסט בדיקה
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    print('Train set shape:', X_train.shape)
    print('Validation set shape:', X_val.shape)
    
    # נרמול
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # מודל LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        feature_fraction=0.9,
        min_child_samples=1,
        min_data_in_bin=1,
        random_state=42
    )
    if len(X_val) > 0:
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(50)]
        )
    else:
        model.fit(X_train_scaled, y_train)
    
    # הצגת חשיבות פיצ'רים
    lgb.plot_importance(model, max_num_features=10)
    plt.title(f'Feature Importance: {platform} ({period_type})')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{platform}_{period_type}.png')
    plt.close()
    
    return model, scaler, feature_cols

def generate_forecast(model, scaler, features, last_date, platform, period_type):
    """Generate forecast for next 12 periods (month/quarter/year)"""
    if period_type == 'month':
        freq = 'MS'
        periods = 12
    elif period_type == 'quarter':
        freq = 'QS'
        periods = 4
    elif period_type == 'year':
        freq = 'YS'
        periods = 2
    else:
        return None
    future_dates = pd.date_range(start=last_date + pd.offsets.DateOffset(1), periods=periods, freq=freq)
    future_df = pd.DataFrame({'date': future_dates})
    future_df['platform'] = platform
    future_df['period_type'] = period_type
    future_df = create_features(future_df)
    # שמירה רק על הפיצ'רים שהמודל דורש
    missing_features = [f for f in features if f not in future_df.columns]
    if missing_features:
        print(f"Missing features in future_df: {missing_features}")
        for f in missing_features:
            future_df[f] = 0  # או ערך ברירת מחדל
    X_future = future_df[features]
    X_future_scaled = scaler.transform(X_future)
    predictions = model.predict(X_future_scaled)
    future_df['predicted_revenue'] = predictions
    return future_df[['date','platform','predicted_revenue','period_type']]

def save_forecast(forecast: pd.DataFrame, platform: str, period_type: str) -> None:
    filename = f"forecast_{platform.lower()}_{period_type}.parquet"
    table = pa.Table.from_pandas(forecast)
    pq.write_table(table, filename)
    logger.info(f"Forecast saved to {filename}")

class RevenueForecaster:
    def __init__(self, platform):
        self.platform = platform
        self.engine = create_engine(DATABASE_URL)
        self.prophet_model = None
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self):
        """טעינת נתונים היסטוריים"""
        query = """
            SELECT date, revenue_usd
            FROM monthly_revenue_total
            WHERE platform = :platform
            ORDER BY date
        """
        df = pd.read_sql(query, self.engine, params={'platform': self.platform})
        df['date'] = pd.to_datetime(df['date'])
        return df
        
    def prepare_features(self, df):
        """הכנת תכונות למודלים"""
        # תכונות בסיסיות
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        
        # חגים
        us_holidays = holidays.US()
        df['is_holiday'] = df['date'].apply(lambda x: x in us_holidays)
        
        # אירועי מוזיקה
        for event, date in MUSIC_EVENTS.items():
            df[f'is_{event.lower()}'] = df['date'].apply(
                lambda x: abs((x - pd.to_datetime(date)).days) <= 7
            )
            
        # קמפיינים
        # נבנה DataFrame של כל הקמפיינים הפעילים עבור כל חודש
        campaign_query = """
            SELECT start_date, end_date, platform, status
            FROM campaigns
            WHERE platform = :platform
            AND status = 'Active'
        """
        campaigns = pd.read_sql(
            campaign_query,
            self.engine,
            params={'platform': self.platform}
        )
        df['active_campaigns'] = 0
        if not campaigns.empty:
            campaigns['start_date'] = pd.to_datetime(campaigns['start_date'])
            campaigns['end_date'] = pd.to_datetime(campaigns['end_date'])
            for idx, row in df.iterrows():
                date = row['date']
                # סופרים קמפיינים שהיו פעילים בתאריך הזה
                active = ((campaigns['start_date'] <= date) & (campaigns['end_date'] >= date)).sum()
                df.at[idx, 'active_campaigns'] = active

        return df
        
    def train_prophet(self, df):
        """אימון מודל Prophet"""
        prophet_df = df.rename(columns={'date': 'ds', 'revenue_usd': 'y'})
        
        # הוספת חגים
        us_holidays = holidays.US()
        holiday_df = pd.DataFrame([
            {'holiday': name, 'ds': date}
            for date, name in us_holidays.items()
        ])
        
        # הוספת אירועי מוזיקה
        for event, date in MUSIC_EVENTS.items():
            holiday_df = pd.concat([
                holiday_df,
                pd.DataFrame({
                    'holiday': event,
                    'ds': [pd.to_datetime(date)]
                })
            ])
            
        self.prophet_model = Prophet(
            holidays=holiday_df,
            **PROPHET_SETTINGS
        )
        self.prophet_model.fit(prophet_df)
        
    def train_xgboost(self, df):
        """אימון מודל XGBoost"""
        features = df.drop(['date', 'revenue_usd'], axis=1)
        target = df['revenue_usd']
        
        # נרמול תכונות
        self.feature_columns = features.columns.tolist()
        features_scaled = self.scaler.fit_transform(features)
        
        self.xgb_model = xgb.XGBRegressor(**XGBOOST_SETTINGS)
        self.xgb_model.fit(features_scaled, target)
        
    def train_lstm(self, df):
        """אימון מודל LSTM"""
        features = df.drop(['date', 'revenue_usd'], axis=1)
        target = df['revenue_usd']
        
        # נרמול תכונות
        self.feature_columns = features.columns.tolist()
        features_scaled = self.scaler.fit_transform(features)
        target_scaled = self.target_scaler.fit_transform(target.values.reshape(-1, 1))
        
        # הכנת נתונים ל-LSTM
        X = []
        y = []
        for i in range(len(features_scaled) - 12):
            X.append(features_scaled[i:(i + 12)])
            y.append(target_scaled[i + 12])
        X = np.array(X)
        y = np.array(y)
        
        # בניית מודל
        self.lstm_model = Sequential([
            LSTM(LSTM_SETTINGS['units'], input_shape=(12, X.shape[2])),
            Dropout(LSTM_SETTINGS['dropout']),
            Dense(1)
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='mse')
        self.lstm_model.fit(
            X, y,
            epochs=LSTM_SETTINGS['epochs'],
            batch_size=LSTM_SETTINGS['batch_size'],
            verbose=0
        )
        
    def forecast(self, periods=12):
        """תחזית הכנסות"""
        df = self.load_data()
        
        # ניקוי NaN בתאריכים לפני המשך עבודה
        if df['date'].isna().any():
            logger.error('יש ערכים חסרים (NaN) בעמודת date בנתונים ההיסטוריים!')
            print(df[df['date'].isna()])
            raise ValueError('ערכים חסרים בעמודת date - יש לתקן את הנתונים')
        
        # המשך רגיל
        df = self.prepare_features(df)
        
        # אימון המודלים
        self.train_prophet(df)
        self.train_xgboost(df)
        self.train_lstm(df)
        
        # תחזית Prophet
        future_dates = pd.date_range(
            start=df['date'].max() + timedelta(days=1),
            periods=periods,
            freq='M'
        )
        logger.info(f"future_dates: {future_dates}")
        if pd.isna(future_dates).any():
            logger.error('יש ערכים חסרים (NaN) ב-future_dates!')
            print(future_dates)
            raise ValueError('ערכים חסרים ב-future_dates - יש לתקן את הקוד')
        prophet_forecast = self.prophet_model.predict(
            pd.DataFrame({'ds': future_dates})
        )
        if prophet_forecast['ds'].isna().any():
            logger.error('יש ערכים חסרים (NaN) בעמודת ds בתחזית Prophet!')
            print(prophet_forecast[prophet_forecast['ds'].isna()])
            raise ValueError('ערכים חסרים בעמודת ds בתחזית Prophet - יש לתקן את הקוד')
        
        # תחזית XGBoost
        future_features = self.prepare_features(
            pd.DataFrame({'date': future_dates})
        )
        # התאמה לעמודות הפיצ'רים
        future_features = future_features[self.feature_columns]
        future_features_scaled = self.scaler.transform(future_features)
        xgb_forecast = self.xgb_model.predict(future_features_scaled)
        
        # תחזית LSTM
        lstm_input = []
        for i in range(len(future_features_scaled) - 12):
            lstm_input.append(future_features_scaled[i:(i + 12)])
        lstm_input = np.array(lstm_input)
        if len(lstm_input) > 0:
            lstm_forecast = self.lstm_model.predict(lstm_input)
            lstm_forecast = self.target_scaler.inverse_transform(lstm_forecast)
            lstm_forecast = lstm_forecast.flatten()
            # השלמת ערכים חסרים אם צריך
            if len(lstm_forecast) < len(future_dates):
                lstm_forecast = np.concatenate([
                    np.full(len(future_dates) - len(lstm_forecast), np.nan),
                    lstm_forecast
                ])
        else:
            lstm_forecast = np.full(len(future_dates), np.nan)
        
        # שילוב התחזיות
        combined_forecast = pd.DataFrame({
            'date': future_dates,
            'prophet_forecast': prophet_forecast['yhat'],
            'xgb_forecast': xgb_forecast,
            'lstm_forecast': lstm_forecast,
            'platform': self.platform
        })
        
        # חישוב ממוצע משוקלל
        combined_forecast['forecast'] = (
            combined_forecast['prophet_forecast'] * 0.5 +
            combined_forecast['xgb_forecast'] * 0.3 +
            combined_forecast['lstm_forecast'] * 0.2
        )
        
        return combined_forecast
        
    def save_models(self):
        """שמירת המודלים"""
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # שמירת Prophet
        if self.prophet_model:
            joblib.dump(
                self.prophet_model,
                models_dir / f'prophet_{self.platform}.joblib'
            )
            
        # שמירת XGBoost
        if self.xgb_model:
            joblib.dump(
                self.xgb_model,
                models_dir / f'xgb_{self.platform}.joblib'
            )
            
        # שמירת LSTM
        if self.lstm_model:
            self.lstm_model.save(
                models_dir / f'lstm_{self.platform}.h5'
            )
            
        # שמירת Scaler
        joblib.dump(
            self.scaler,
            models_dir / f'scaler_{self.platform}.joblib'
        )

def main():
    # Clear the forecasts table before running the forecast
    with create_engine(DATABASE_URL).connect() as conn:
        conn.execute(text("DELETE FROM forecasts"))
        conn.commit()
        print("Forecasts table cleared successfully.")

    # Process distribution statements
    distribution_data = process_distribution_statements()
    print("Distribution data processed successfully.")

    # Load all data
    df = load_data()
    # קבלת כל הערכים הייחודיים
    platforms = df['platform'].unique()
    tracks = df['track_id'].unique()
    countries = df['country'].unique()
    period_types = df['period_type'].unique() if 'period_type' in df.columns else ['month']

    for platform in platforms:
        for track_id in tracks:
            for country in countries:
                for period_type in period_types:
                    try:
                        # סינון לפי פילוחים
                        data = df[(df['platform'] == platform) & (df['track_id'] == track_id) & (df['country'] == country) & (df['period_type'] == period_type)]
                        if len(data) < 6:
                            continue  # לא מספיק נתונים לתחזית
                        model, scaler, feature_cols = train_model(data, platform, period_type)
                        if model is None:
                            continue
                        last_date = data['date'].max()
                        forecast = generate_forecast(model, scaler, feature_cols, last_date, platform, period_type)
                        forecast['track_id'] = track_id
                        forecast['country'] = country
                        forecast['period_type'] = period_type
                        # שמירת התחזית
                        engine = create_engine(DATABASE_URL)
                        forecast.to_sql('forecasts', engine, if_exists='append', index=False)
                        logger.info(f"Successfully forecasted {platform} | {track_id} | {country} | {period_type}")
                    except Exception as e:
                        logger.error(f"Error forecasting {platform} | {track_id} | {country} | {period_type}: {str(e)}")
                        continue

if __name__ == '__main__':
    main() 