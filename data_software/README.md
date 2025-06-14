# Streaming Revenue Analysis

A system for analyzing and forecasting streaming revenue for a music label.

## Key Features

- Automatic processing of streaming reports (CSV/Excel)
- Conversion to daily time series
- Revenue forecasting using Prophet
- Interactive dashboard with Streamlit
- Automatic refresh when new files are added

## System Requirements

- Python 3.11 or higher
- pip (Python package manager)

## Installation

1. Install Python 3.11:

   ```bash
   # macOS
   brew install python@3.11
   ```

2. Create virtual environment and install dependencies:

   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate  # Windows

   pip install -r requirements.txt
   ```

3. Set up environment file:
   ```bash
   cp .env.example .env
   # Edit .env file to adjust platform payout rates
   ```

## Usage

1. Start the file monitoring system:

   ```bash
   python watch_folder.py
   ```

2. Launch the dashboard:

   ```bash
   streamlit run dashboard.py
   ```

3. Copy your report files to the `data_raw/` directory

## Project Structure

- `etl.py` - Data processing
- `run_forecast.py` - Revenue forecasting
- `dashboard.py` - Interactive dashboard
- `watch_folder.py` - File monitoring
- `config.py` - System configuration
- `data_raw/` - Report files directory
- `mvp.db` - SQLite database
- `forecast_*.parquet` - Forecast files

## Report File Format

Report files should include the following columns:

- `date` (YYYY-MM-DD or DD/MM/YYYY)
- `track_id` / `isrc`
- `platform` (Spotify, Apple Music, YouTube, Beatport...)
- `streams` (or `downloads`)
- `revenue_usd` (optional)

## Notes

- The system will automatically calculate revenue if the `revenue_usd` column is missing
- Forecasts are generated for 90 days ahead
- You can view forecasts for the entire catalog or individual tracks
