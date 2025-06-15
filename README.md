# On The Way Records - Revenue Forecast System

This system provides revenue forecasting and analysis for On The Way Records' music distribution data.

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create required directories:

```bash
mkdir -p input processed data
```

## Usage

1. Place your distribution statement CSV files in the `input` directory.

2. Run the ETL process:

```bash
python data_software/etl.py
```

3. Launch the dashboard:

```bash
streamlit run data_software/dashboard.py
```

## Features

- Revenue forecasting for Spotify, Beatport, and Overall
- Historical data analysis
- Platform comparison
- Monthly revenue trends
- Forecast accuracy metrics

## Data Format

The system expects CSV files with the following columns:

- For Create Music Group format: `store/platform`, `month`, `recipient net royalty ($ usd)`
- For Amp format: `distributor/label`, `period from/transaction date`, `revenue`

## Directory Structure

- `input/`: Place your distribution statement CSV files here
- `processed/`: Processed data files
- `data/`: Database and other data files
- `data_software/`: Source code
  - `etl.py`: Data processing and ETL
  - `dashboard.py`: Streamlit dashboard
  - `config.py`: Configuration settings
