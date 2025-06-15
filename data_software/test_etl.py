import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import os
import sqlite3
import shutil
from etl import (
    map_columns,
    normalize_dates,
    process_revenue,
    fill_missing_dates,
    process_file,
    detect_file_type,
    process_pivot_file,
    process_regular_file,
    init_db
)

class TestETL(unittest.TestCase):
    def setUp(self):
        # יצירת נתוני בדיקה
        self.test_data = pd.DataFrame({
            'transaction date': ['2024-01-01', '2024-02-01'],
            'isrc': ['ISRC1', 'ISRC2'],
            'distributor': ['Spotify', 'Apple Music'],
            'quantity': [1000, 2000],
            'revenue': [10.0, 20.0]
        })
        
        # יצירת תיקיית בדיקה
        self.test_dir = Path('test_data')
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(exist_ok=True)
        
        # יצירת קובץ בדיקה
        self.test_file = self.test_dir / 'test_report.csv'
        self.test_data.to_csv(self.test_file, index=False)

    def tearDown(self):
        # ניקוי קבצי בדיקה
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_map_columns(self):
        """בדיקת מיפוי עמודות"""
        result = map_columns(self.test_data)
        expected_columns = ['date', 'track_id', 'platform', 'streams', 'revenue_usd']
        self.assertEqual(list(result.columns), expected_columns)

    def test_normalize_dates(self):
        """בדיקת נרמול תאריכים"""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2424-02-01', 'invalid']
        })
        result = normalize_dates(df)
        self.assertEqual(len(result), 2)  # רק שני תאריכים תקינים
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['date']))

    def test_process_revenue(self):
        """בדיקת חישוב הכנסות"""
        df = pd.DataFrame({
            'platform': ['Spotify', 'Apple Music'],
            'streams': [1000, 2000]
        })
        result = process_revenue(df)
        self.assertIn('revenue_usd', result.columns)
        self.assertTrue(result['revenue_usd'].notna().all())

    def test_fill_missing_dates(self):
        """בדיקת מילוי תאריכים חסרים"""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-02-01']),
            'platform': ['Spotify', 'Spotify'],
            'track_id': ['ISRC1', 'ISRC2'],
            'country': ['US', 'US'],
            'revenue_usd': [100, 200]
        })
        result = fill_missing_dates(df)
        self.assertEqual(len(result), 2)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['date']))

    def test_detect_file_type(self):
        """בדיקת זיהוי סוג הקובץ"""
        # יצירת קובץ pivot
        pivot_data = pd.DataFrame({
            'Track': ['Track1', 'Track2'],
            'january 24': [100, 200],
            'february 24': [150, 250]
        })
        pivot_file = self.test_dir / 'pivot_report.csv'
        pivot_data.to_csv(pivot_file, index=False)
        
        # בדיקת זיהוי קובץ pivot
        self.assertEqual(detect_file_type(str(pivot_file)), 'pivot')
        
        # בדיקת זיהוי קובץ רגיל
        self.assertEqual(detect_file_type(str(self.test_file)), 'regular')
        
        # ניקוי
        pivot_file.unlink()

    def test_process_pivot_file(self):
        """בדיקת עיבוד קובץ pivot"""
        # יצירת קובץ pivot
        pivot_data = pd.DataFrame({
            'Track': ['Track1', 'Track2'],
            'Platform': ['Spotify', 'Apple Music'],
            'january 24': [100, 200],
            'february 24': [150, 250]
        })
        pivot_file = self.test_dir / 'pivot_report.csv'
        pivot_data.to_csv(pivot_file, index=False)
        
        # בדיקת עיבוד
        result = process_pivot_file(str(pivot_file))
        self.assertIn('date', result.columns)
        self.assertIn('revenue_usd', result.columns)
        self.assertIn('period_type', result.columns)
        
        # ניקוי
        pivot_file.unlink()

    def test_process_regular_file(self):
        """בדיקת עיבוד קובץ רגיל"""
        # יצירת קובץ CSV במקום Excel
        regular_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-02-01'],
            'track_id': ['ISRC1', 'ISRC2'],
            'platform': ['Spotify', 'Apple Music'],
            'revenue_usd': [100, 200]
        })
        regular_file = self.test_dir / 'regular_report.csv'
        regular_data.to_csv(regular_file, index=False)
        
        result = process_regular_file(str(regular_file))
        self.assertIn('date', result.columns)
        self.assertIn('revenue_usd', result.columns)
        
        # ניקוי
        regular_file.unlink()

    def test_database_operations(self):
        """בדיקת פעולות בסיסיות על בסיס הנתונים"""
        # יצירת בסיס נתונים זמני
        test_db = 'test.db'
        if os.path.exists(test_db):
            os.remove(test_db)
        
        # בדיקת יצירת טבלאות
        init_db()
        
        # בדיקת חיבור לבסיס הנתונים
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        
        # בדיקת קיום הטבלאות
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        expected_tables = ['monthly_revenue_total', 'campaigns', 'costs', 'artists', 'releases', 'platform_metrics', 'forecasts']
        self.assertTrue(all(table[0] in expected_tables for table in tables))
        
        # ניקוי
        conn.close()
        os.remove(test_db)

if __name__ == '__main__':
    unittest.main() 