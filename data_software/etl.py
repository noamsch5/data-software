from typing import List, Optional
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import logging
from config import PLATFORM_PAYOUTS, DB_PATH, RAW_DATA_DIR, DATABASE_URL
import calendar
import re
import os
import numpy as np
from sqlalchemy import create_engine, text
import glob
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column names to standard names"""
    column_mapping = {
        # Date
        'transaction date': 'date',
        'month': 'date',
        'period to': 'date',
        # Track ID
        'isrc': 'track_id',
        # Platform
        'distributor': 'platform',
        'store': 'platform',
        # Streams/Downloads/Quantity
        'quantity': 'streams',
        'downloads': 'streams',
        # Revenue
        'revenue': 'revenue_usd',
        'recipient net royalty ($ usd)': 'revenue_usd',
    }
    # lowercase columns for matching
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    return df

def read_streaming_file(file_path: Path) -> pd.DataFrame:
    """Read streaming file (CSV or Excel)"""
    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Print original columns for debug
        logger.warning(f"File {file_path} original columns: {list(df.columns)}")
        
        # Map columns to standard names
        df = map_columns(df)
        # Print columns after mapping for debug
        logger.warning(f"File {file_path} columns after mapping: {list(df.columns)}")
        
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Skip files without ISRC/track_id and streams/quantity
        if not (('track_id' in df.columns or 'isrc' in df.columns) and ('streams' in df.columns or 'quantity' in df.columns)):
            logger.warning(f"File {file_path} skipped: missing ISRC/track_id or streams/quantity column")
            return pd.DataFrame()
        
        # If 'isrc' exists but not 'track_id', rename
        if 'isrc' in df.columns and 'track_id' not in df.columns:
            df = df.rename(columns={'isrc': 'track_id'})
        # If 'quantity' exists but not 'streams', rename
        if 'quantity' in df.columns and 'streams' not in df.columns:
            df = df.rename(columns={'quantity': 'streams'})
        # If 'store' exists but not 'platform', rename
        if 'store' in df.columns and 'platform' not in df.columns:
            df = df.rename(columns={'store': 'platform'})
        # If 'distributor' exists but not 'platform', rename
        if 'distributor' in df.columns and 'platform' not in df.columns:
            df = df.rename(columns={'distributor': 'platform'})
        # If 'month' or 'period to' exists but not 'date', rename
        if 'month' in df.columns and 'date' not in df.columns:
            df = df.rename(columns={'month': 'date'})
        if 'period to' in df.columns and 'date' not in df.columns:
            df = df.rename(columns={'period to': 'date'})
        # If 'recipient net royalty ($ usd)' exists but not 'revenue_usd', rename
        if 'recipient net royalty ($ usd)' in df.columns and 'revenue_usd' not in df.columns:
            df = df.rename(columns={'recipient net royalty ($ usd)': 'revenue_usd'})
        if 'revenue' in df.columns and 'revenue_usd' not in df.columns:
            df = df.rename(columns={'revenue': 'revenue_usd'})
        
        # Verify required columns
        required_columns = ['date', 'track_id', 'platform', 'streams']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return pd.DataFrame()

def process_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate revenue if not present"""
    if 'revenue_usd' not in df.columns:
        df['revenue_usd'] = df.apply(
            lambda row: row['streams'] * PLATFORM_PAYOUTS.get(row['platform'].lower(), 0),
            axis=1
        )
    return df

def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dates to ISO format, fix out-of-bounds years, and drop invalid dates"""
    # Print unique date values for debug
    logger.warning(f"Unique date values before normalization: {df['date'].unique()}")
    # Try to fix years like 2424 -> 2024
    def fix_year(val):
        try:
            s = str(val)
            if len(s) >= 4 and s[:2] == '24' and int(s[:4]) > 2100:
                return '20' + s[2:]
            return val
        except:
            return val
    df['date'] = df['date'].apply(fix_year)
    # Convert to datetime, coerce errors
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Drop rows with invalid dates
    before = len(df)
    df = df.dropna(subset=['date'])
    after = len(df)
    if before != after:
        logger.warning(f"Dropped {before-after} rows with invalid dates")
    return df

def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing dates with 0 revenue, group by month, platform, track_id, country, title, and create a 'date' column (YYYY-MM-01)"""
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    # Group by month, platform, track_id, country, title, sum revenue
    group_cols = ['month', 'platform']
    if 'track_id' in df.columns:
        group_cols.append('track_id')
    if 'country' in df.columns:
        group_cols.append('country')
    if 'title' in df.columns:
        group_cols.append('title')
    grouped = df.groupby(group_cols)['revenue_usd'].sum().reset_index()
    grouped['date'] = grouped['month']
    # סדר עמודות
    cols = ['date', 'platform', 'revenue_usd']
    if 'track_id' in grouped.columns:
        cols.append('track_id')
    if 'country' in grouped.columns:
        cols.append('country')
    if 'title' in grouped.columns:
        cols.append('title')
    grouped = grouped[cols]
    return grouped

def update_database(df: pd.DataFrame) -> None:
    """Update database with all relevant columns"""
    print(f"update_database: df.shape={df.shape}, columns={df.columns.tolist()}")
    try:
        conn = sqlite3.connect(DB_PATH)
        df.to_sql('monthly_revenue_total', conn, if_exists='replace', index=False)
        conn.close()
        logger.info("Database updated successfully")
    except Exception as e:
        logger.error(f"Error updating database: {str(e)}")
        raise

def process_new_file(file_path: Path) -> None:
    """Process new file"""
    try:
        logger.info(f"Processing new file: {file_path}")
        df = read_streaming_file(file_path)
        df = process_revenue(df)
        df = normalize_dates(df)
        df = fill_missing_dates(df)
        update_database(df)
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise

def detect_file_type(file_path: str) -> str:
    """Detect if file is pivot or regular report (תמיכה ב-CSV/Excel)"""
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    # בדיקה אם יש עמודות של חודשים או רבעונים
    month_columns = [col for col in df.columns if any(month in str(col).lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec','january','february','march','april','may','june','july','august','september','october','november','december'])]
    quarter_columns = [col for col in df.columns if "q" in str(col).lower() and "'" in str(col)]
    if month_columns or quarter_columns:
        return 'pivot'
    return 'regular'

def process_pivot_file(file_path: str) -> pd.DataFrame:
    """Process pivot file (CSV/Excel) with monthly/quarterly/yearly/overall data and period_type"""
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    df.columns = [c.lower().strip() for c in df.columns]
    month_names = ['january','february','march','april','may','june','july','august','september','october','november','december']
    # עמודות חודשים: תומך גם בפורמט 'january 25' וכו'
    month_regex = r'^(january|february|march|april|may|june|july|august|september|october|november|december) ?\'?\d{2,4}$'
    month_columns = [col for col in df.columns if re.match(month_regex, str(col).lower())]
    # עמודות רבעון: Q1/Q2/Q3/Q4 + שנה (למשל Q2 '24 או Q2 24)
    quarter_regex = r"^q[1-4] ?'?\d{2,4}$"
    quarter_columns = [col for col in df.columns if re.match(quarter_regex, str(col).lower())]
    year_columns = [col for col in df.columns if re.match(r"^20\d{2}$", str(col))]
    overall_columns = [col for col in df.columns if 'overall' in str(col).lower() or 'total' in str(col).lower()]
    dfs = []
    if month_columns:
        df_month = pd.melt(df, id_vars=[col for col in df.columns if col not in month_columns],
                          value_vars=month_columns, var_name='period', value_name='revenue_usd')
        df_month = df_month.reset_index(drop=True)
        df_month['period_type'] = 'month'
        month_map = {'january':'01','february':'02','march':'03','april':'04','may':'05','june':'06','july':'07','august':'08','september':'09','october':'10','november':'11','december':'12'}
        df_month['month_name'] = df_month['period'].str.extract(r'^(january|february|march|april|may|june|july|august|september|october|november|december)', flags=re.IGNORECASE)[0].str.lower()
        df_month['month_num'] = df_month['month_name'].map(month_map)
        df_month['year'] = df_month['period'].str.extract(r'(\d{2,4})')[0]
        df_month['year'] = df_month['year'].apply(lambda x: '20'+x if len(x)==2 else x)
        df_month['date'] = pd.to_datetime(df_month['year'] + '-' + df_month['month_num'] + '-01', errors='coerce')
        df_month = df_month.drop_duplicates(subset=['platform', 'date', 'revenue_usd'])
        dfs.append(df_month)
    if quarter_columns:
        df_quarter = pd.melt(df, id_vars=[col for col in df.columns if col not in quarter_columns],
                          value_vars=quarter_columns, var_name='period', value_name='revenue_usd')
        df_quarter = df_quarter.reset_index(drop=True)
        df_quarter['period_type'] = 'quarter'
        df_quarter['year'] = df_quarter['period'].str.extract(r'(\d{2,4})').fillna('2024')
        df_quarter['year'] = df_quarter['year'].apply(lambda x: '20'+x if len(x)==2 else x)
        quarter_map = {"q1": "03-31", "q2": "06-30", "q3": "09-30", "q4": "12-31"}
        df_quarter['quarter_num'] = df_quarter['period'].str.lower().str.extract(r'(q[1-4])')[0]
        df_quarter['date'] = pd.to_datetime(df_quarter['year'] + '-' + df_quarter['quarter_num'].map(quarter_map), errors='coerce')
        df_quarter = df_quarter.drop_duplicates(subset=['platform', 'date', 'revenue_usd'])
        dfs.append(df_quarter)
    if year_columns:
        df_year = pd.melt(df, id_vars=[col for col in df.columns if col not in year_columns],
                          value_vars=year_columns, var_name='period', value_name='revenue_usd')
        df_year = df_year.reset_index(drop=True)
        df_year['period_type'] = 'year'
        df_year['date'] = pd.to_datetime(df_year['period'] + '-12-31', errors='coerce')
        df_year = df_year.drop_duplicates(subset=['platform', 'date', 'revenue_usd'])
        dfs.append(df_year)
    if overall_columns:
        df_overall = pd.melt(df, id_vars=[col for col in df.columns if col not in overall_columns],
                          value_vars=overall_columns, var_name='period', value_name='revenue_usd')
        df_overall = df_overall.reset_index(drop=True)
        df_overall['period_type'] = 'overall'
        df_overall['date'] = pd.NaT
        df_overall = df_overall.groupby(['platform','period_type'], as_index=False).agg({'revenue_usd':'sum','date':'first'})
        dfs.append(df_overall)
    # איחוד הכל
    df_long = pd.concat(dfs, ignore_index=True)
    # ניקוי ערכים
    df_long['revenue_usd'] = pd.to_numeric(df_long['revenue_usd'].replace('[\$,]', '', regex=True), errors='coerce')
    df_long = df_long.dropna(subset=['revenue_usd'])
    df_long['platform'] = df_long['platform'].fillna('Overall')
    # שמירה רק על עמודות רלוונטיות
    result_df = df_long[['date','platform','revenue_usd','period_type']]
    # דיבאג
    for ptype in result_df['period_type'].unique():
        print(f"period_type={ptype}: rows={len(result_df[result_df['period_type']==ptype])}, sum={result_df[result_df['period_type']==ptype]['revenue_usd'].sum()}")
    return result_df

def process_regular_file(file_path: str) -> pd.DataFrame:
    """Process regular report file"""
    # בדיקת סוג הקובץ
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    
    # מיפוי עמודות
    column_mapping = {
        'date': ['date', 'period', 'month'],
        'track_id': ['track_id', 'isrc', 'track'],
        'platform': ['platform', 'store', 'service'],
        'revenue_usd': ['revenue', 'revenue_usd', 'amount']
    }
    
    # זיהוי עמודות
    mapped_columns = {}
    for target, possible_names in column_mapping.items():
        for col in df.columns:
            if any(name.lower() in str(col).lower() for name in possible_names):
                mapped_columns[target] = col
                break
    
    # המרה לפורמט הנדרש
    result_df = pd.DataFrame()
    for target, source in mapped_columns.items():
        result_df[target] = df[source]
    
    # הוספת עמודת פלטפורמה אם חסרה
    if 'platform' not in result_df.columns:
        result_df['platform'] = 'Overall'
    
    return result_df

def process_file(file_path: str) -> None:
    """Process streaming report file"""
    try:
        file_type = detect_file_type(file_path)
        if file_type == 'pivot':
            df = process_pivot_file(file_path)
        else:
            df = process_regular_file(file_path)
        print(f"process_file: after processing {file_path}, df.shape={df.shape}, columns={df.columns.tolist()}")
        update_database(df)
        logger.info(f"File {file_path} processed successfully")
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise

def main():
    """Main function to process all files in the input directory using only process_distribution_statements."""
    init_db()  # Create all required tables
    df = process_distribution_statements()
    print("Unified distribution data (first 10 rows):")
    print(df.head(10))

    # שלב תחזית: המודל לומד את הנתונים ומבצע תחזית לכל פלטפורמה
    try:
        from prophet import Prophet
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from sqlalchemy import create_engine

        engine = create_engine(DATABASE_URL)
        # טען את טבלת monthly_revenue_total
        monthly = pd.read_sql('SELECT * FROM monthly_revenue_total', engine)
        platforms = monthly['platform'].unique()
        all_forecasts = []
        for platform in platforms:
            df_platform = monthly[monthly['platform'] == platform].sort_values('date')
            if len(df_platform) < 3:
                continue
            df_agg = df_platform.groupby('date')['revenue_usd'].sum().reset_index()
            df_agg.columns = ['ds', 'y']
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
            model.fit(df_agg)
            future = model.make_future_dataframe(periods=12, freq='M')
            forecast = model.predict(future)
            forecast_out = pd.DataFrame({
                'date': forecast['ds'],
                'platform': platform,
                'period_type': 'month',
                'predicted_revenue': forecast['yhat']
            })
            # שמירה לקובץ parquet
            table = pa.Table.from_pandas(forecast_out)
            pq.write_table(table, f"forecast_{platform}.parquet")
            all_forecasts.append(forecast_out)
        # איחוד כל התחזיות ושמירה ל-DB
        if all_forecasts:
            all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
            all_forecasts_df.to_sql('forecasts', engine, if_exists='replace', index=False)
            print("התחזיות נשמרו בהצלחה בטבלה forecasts ובקבצי parquet.")
        else:
            print("לא נוצרו תחזיות (אין מספיק נתונים לכל פלטפורמה)")
    except Exception as e:
        print(f"שגיאה בביצוע התחזית: {e}")

def init_db():
    """Create the database and tables only, without inserting sample data."""
    engine = create_engine(DATABASE_URL)
    
    # הגדרת הטבלאות
    tables = {
        'monthly_revenue_total': """
            CREATE TABLE IF NOT EXISTS monthly_revenue_total (
                date DATE,
                platform TEXT,
                revenue_usd REAL,
                period_type TEXT,
                country TEXT,
                track_id TEXT,
                title TEXT,
                PRIMARY KEY (date, platform, country, track_id)
            )
        """,
        'campaigns': """
            CREATE TABLE IF NOT EXISTS campaigns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                start_date DATE,
                end_date DATE,
                budget REAL,
                platform TEXT,
                type TEXT,
                status TEXT
            )
        """,
        'costs': """
            CREATE TABLE IF NOT EXISTS costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                category TEXT,
                amount REAL,
                description TEXT,
                campaign_id INTEGER,
                FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
            )
        """,
        'artists': """
            CREATE TABLE IF NOT EXISTS artists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                genre TEXT,
                country TEXT
            )
        """,
        'releases': """
            CREATE TABLE IF NOT EXISTS releases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artist_id INTEGER,
                title TEXT,
                type TEXT,
                release_date DATE,
                platform TEXT,
                FOREIGN KEY (artist_id) REFERENCES artists(id)
            )
        """,
        'platform_metrics': """
            CREATE TABLE IF NOT EXISTS platform_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                platform TEXT,
                metric_name TEXT,
                metric_value REAL
            )
        """,
        'forecasts': """
            CREATE TABLE IF NOT EXISTS forecasts (
                date DATE,
                platform TEXT,
                forecast REAL,
                prophet_forecast REAL,
                xgb_forecast REAL,
                lstm_forecast REAL,
                period_type TEXT,
                PRIMARY KEY (date, platform)
            )
        """
    }
    
    # יצירת הטבלאות
    for table_name, create_sql in tables.items():
        try:
            with engine.connect() as conn:
                conn.execute(text(create_sql))
                conn.commit()
                logger.info(f"Table {table_name} created successfully")
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {str(e)}")

def process_distribution_statements():
    """
    Process distribution statements from the input folder.
    Supports both Create Music Group and Amp file formats.
    Detects file type by columns, maps the correct columns, and outputs a unified DataFrame.
    """
    input_dirs = ['input', 'data_software/input']
    all_data = []
    found_files = []
    for input_dir in input_dirs:
        files = glob.glob(os.path.join(input_dir, '*.csv'))
        found_files.extend(files)
    print(f"Found {len(found_files)} input files: {found_files}")
    if not found_files:
        print("No input files found in 'input/' or 'data_software/input/'. Please upload at least one CSV file.")
        return pd.DataFrame()
    for file_path in found_files:
        print(f"\n---\nProcessing file: {file_path}")
        try:
            df = pd.read_csv(file_path)
            print(f"Columns in {file_path}: {list(df.columns)}")
            print(f"First row: {df.iloc[0].to_dict() if not df.empty else '(empty)'}")
            # Convert column names to lowercase
            df.columns = [str(c).strip().lower() for c in df.columns]
            columns = df.columns
            print(f'Columns: {list(df.columns)}')
            print(f'First row: {df.iloc[0].to_dict() if not df.empty else "(empty)"}')
            
            # Detect Create Music Group format
            if 'recipient net royalty ($ usd)' in columns:
                print('Detected format: Create Music Group')
                df_unified = pd.DataFrame({
                    'Platform': df['store'] if 'store' in columns else df['platform'],
                    'Month': pd.to_datetime(df['month']).dt.to_period('M').astype(str),
                    'Revenue': df['recipient net royalty ($ usd)']
                })
            # Detect Amp format
            elif 'revenue' in columns:
                print('Detected format: Amp')
                revenue = df['revenue']
                if 'amp' in file_path.lower():
                    revenue = revenue * 1.35
                # Use Period From as month if exists
                if 'period from' in columns:
                    month = pd.to_datetime(df['period from'], errors='coerce').dt.to_period('M').astype(str)
                elif 'transaction date' in columns:
                    month = pd.to_datetime(df['transaction date'], errors='coerce').dt.to_period('M').astype(str)
                else:
                    month = ''
                df_unified = pd.DataFrame({
                    'Platform': df['distributor'] if 'distributor' in columns else (df['label'] if 'label' in columns else ''),
                    'Month': month,
                    'Revenue': revenue
                })
            else:
                print(f"Unrecognized file! columns: {df.columns}")
                print(f"Required column names: recipient net royalty ($ usd) or revenue")
                continue
            all_data.append(df_unified)
        except Exception as e:
            print(f'Error reading file {file_path}: {e}')

    if not all_data:
        print("No valid distribution data found.")
        return pd.DataFrame()

    combined_data = pd.concat(all_data, ignore_index=True)

    # Aggregate by Platform and Month
    aggregated_data = combined_data.groupby([
        'Platform', 'Month'
    ]).agg({
        'Revenue': 'sum'
    }).reset_index()

    # Save full data (for additional analysis)
    try:
        engine = create_engine(DATABASE_URL)
        aggregated_data.to_sql('distribution_full', engine, if_exists='replace', index=False)
        print("Full data saved successfully to distribution_full table.")
    except Exception as e:
        print(f"Error saving full data: {e}")

    # Create monthly_revenue_total table in the format expected by the dashboard
    try:
        monthly = aggregated_data.copy()
        monthly['date'] = pd.to_datetime(monthly['Month'] + '-01', errors='coerce')
        monthly['platform'] = monthly['Platform']
        monthly['revenue_usd'] = monthly['Revenue']
        monthly['period_type'] = 'month'
        if 'Country' in monthly.columns:
            monthly['country'] = monthly['Country']
        if 'Track Title' in monthly.columns:
            monthly['title'] = monthly['Track Title']
        if 'ISRC' in monthly.columns:
            monthly['track_id'] = monthly['ISRC']
        # Add Overall platform
        overall = monthly.groupby('date').agg({
            'revenue_usd': 'sum'
        }).reset_index()
        overall['platform'] = 'Overall'
        overall['period_type'] = 'month'
        # Combine platform-specific and overall data
        monthly_revenue_total = pd.concat([
            monthly[['date', 'platform', 'revenue_usd', 'period_type', 'country', 'track_id', 'title']].copy(),
            overall[['date', 'platform', 'revenue_usd', 'period_type']].copy()
        ], ignore_index=True)
        monthly_revenue_total.to_sql('monthly_revenue_total', engine, if_exists='replace', index=False)
        print("Data saved successfully to monthly_revenue_total table.")
    except Exception as e:
        print(f"Error saving monthly_revenue_total table: {e}")

    return aggregated_data

if __name__ == '__main__':
    main() 