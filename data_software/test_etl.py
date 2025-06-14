import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from etl import (
    map_columns,
    normalize_dates,
    process_revenue,
    fill_missing_dates
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
            'revenue_usd': [100, 200]
        })
        result = fill_missing_dates(df)
        self.assertEqual(len(result), 2)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['date']))

if __name__ == '__main__':
    unittest.main() 