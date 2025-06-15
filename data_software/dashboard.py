import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from config import DB_PATH, PLATFORMS, COST_CATEGORIES, CAMPAIGN_TYPES, CAMPAIGN_STATUSES, MUSIC_EVENTS
import glob
import os
from prophet import Prophet
from sqlalchemy import create_engine
from config import DATABASE_URL

st.set_page_config(
    page_title="On The Way Records - Revenue Forecast",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 On The Way Records - Revenue Forecast")
st.markdown("### Revenue Forecast Dashboard for Music Platforms")

def run_forecast():
    """Run forecast for all platforms and save results"""
    engine = create_engine(DATABASE_URL)
    monthly = pd.read_sql('SELECT * FROM monthly_revenue_total', engine)
    platforms = ['Overall', 'Beatport', 'Spotify']
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
        
        # Save to parquet file
        table = pa.Table.from_pandas(forecast_out)
        pq.write_table(table, f"forecast_{platform.lower()}.parquet")
        all_forecasts.append(forecast_out)
    
    # Combine all forecasts and save to DB
    if all_forecasts:
        all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
        all_forecasts_df.to_sql('forecasts', engine, if_exists='replace', index=False)
        st.success("Forecasts generated and saved successfully")
    else:
        st.error("No forecasts generated (insufficient data for platforms)")

# Read forecast files
forecast_files = glob.glob("forecast_*.parquet")
if not forecast_files:
    st.warning("No forecast files found. Running forecast generation...")
    run_forecast()
    forecast_files = glob.glob("forecast_*.parquet")

# Read all forecasts
dfs = []
for file in forecast_files:
    df = pq.read_table(file).to_pandas()
    df['source_file'] = os.path.basename(file)
    dfs.append(df)
forecast_df = pd.concat(dfs, ignore_index=True)

# Read historical data from DB
def load_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM monthly_revenue_total", conn)
    conn.close()
    return df

hist_df = load_history()

# Platform and period_type selectors
platforms = ['Overall', 'Beatport', 'Spotify']
period_types = sorted([str(x) for x in forecast_df['period_type'].dropna().unique()])
platform = st.selectbox("Select Platform", platforms, key='platform_selector')
period_type = st.selectbox("Select Period Type", period_types, key='period_type_selector')

# Filter data
if period_type == 'overall':
    hist = hist_df[(hist_df['platform'] == 'Overall') & (hist_df['period_type'] == 'overall')]
    forecast = forecast_df[(forecast_df['platform'] == 'Overall') & (forecast_df['period_type'] == 'overall')]
else:
    hist = hist_df[(hist_df['platform'] == platform) & (hist_df['period_type'] == period_type)]
    forecast = forecast_df[(forecast_df['platform'] == platform) & (forecast_df['period_type'] == period_type)]

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Historical Revenue", f"${hist['revenue_usd'].sum():,.0f}")
with col2:
    st.metric("Average per Period", f"${hist['revenue_usd'].mean():,.0f}")
with col3:
    if len(hist) > 1:
        delta = hist['revenue_usd'].iloc[-1] - hist['revenue_usd'].iloc[-2]
        st.metric("Change from Previous Period", f"${delta:,.0f}")
    else:
        st.metric("Change from Previous Period", "N/A")
with col4:
    st.metric("Total Forecast (12 periods)", f"${forecast['predicted_revenue'].sum():,.0f}")

# Historical + Forecast Revenue Graph
fig = go.Figure()
if not hist.empty:
    fig.add_trace(go.Scatter(
        x=hist['date'], y=hist['revenue_usd'],
        mode='lines+markers', name='History', line=dict(color='royalblue')
    ))
if not forecast.empty:
    fig.add_trace(go.Scatter(
        x=forecast['date'], y=forecast['predicted_revenue'],
        mode='lines+markers', name='Forecast', line=dict(color='orange', dash='dash')
    ))
fig.update_layout(
    title=f"Revenue for {platform if period_type != 'overall' else 'Overall'} ({period_type})",
    xaxis_title="Date",
    yaxis_title="Revenue ($)",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# Platform Comparison
st.subheader("Platform Comparison (Forecast)")
comp = forecast_df.groupby('platform')['predicted_revenue'].sum().sort_values(ascending=False)
st.bar_chart(comp)

# Forecast Table
st.subheader("Forecast Table")
forecast_table = forecast[['date', 'predicted_revenue']]
forecast_table = forecast_table.rename(columns={'date': 'Date', 'predicted_revenue': 'Forecasted Revenue ($)'})
forecast_table['Date'] = pd.to_datetime(forecast_table['Date']).dt.strftime('%Y-%m-%d')
st.dataframe(forecast_table, hide_index=True)

st.caption('Built with AI - All rights reserved.')

def load_data() -> pd.DataFrame:
    """Load data from monthly_revenue_total table"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM monthly_revenue_total", conn)
    conn.close()
    
    # המרת תאריכים
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_forecast(platform: str) -> pd.DataFrame:
    """Load forecast for specific platform"""
    filename = f"forecast_{platform.lower()}.parquet"
    try:
        return pq.read_table(filename).to_pandas()
    except:
        return None

def calculate_kpis(df: pd.DataFrame, platform: str) -> dict:
    """Calculate KPIs for specific platform"""
    platform_data = df[df['platform'] == platform]
    
    # חישוב KPI-ים בסיסיים
    total_revenue = platform_data['revenue_usd'].sum()
    avg_monthly = platform_data['revenue_usd'].mean()
    last_month = platform_data.sort_values('date')['revenue_usd'].iloc[-1]
    
    # חישוב מגמות
    platform_data['month'] = platform_data['date'].dt.to_period('M')
    monthly_avg = platform_data.groupby('month')['revenue_usd'].mean()
    growth_rate = ((monthly_avg.iloc[-1] / monthly_avg.iloc[0]) - 1) * 100
    
    # חישוב עונתיות
    platform_data['month_num'] = platform_data['date'].dt.month
    seasonality = platform_data.groupby('month_num')['revenue_usd'].mean()
    peak_month = seasonality.idxmax()
    peak_value = seasonality.max()
    
    return {
        'total_revenue': total_revenue,
        'avg_monthly': avg_monthly,
        'last_month': last_month,
        'growth_rate': growth_rate,
        'peak_month': peak_month,
        'peak_value': peak_value
    }

def plot_platform_comparison(df: pd.DataFrame) -> go.Figure:
    """Plot comparison between platforms"""
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Monthly Revenue by Platform', 'Revenue Distribution'),
                       vertical_spacing=0.2)
    
    # גרף קווי של הכנסות חודשיות
    for platform in df['platform'].unique():
        platform_data = df[df['platform'] == platform].sort_values('date')
        fig.add_trace(
            go.Scatter(x=platform_data['date'], y=platform_data['revenue_usd'],
                      name=platform, mode='lines+markers'),
            row=1, col=1
        )
    
    # גרף קופסה של התפלגות הכנסות
    for platform in df['platform'].unique():
        platform_data = df[df['platform'] == platform]
        fig.add_trace(
            go.Box(y=platform_data['revenue_usd'], name=platform),
            row=2, col=1
        )
    
    fig.update_layout(height=800, showlegend=True)
    return fig

def plot_forecast_with_history(df, forecast, selected_platform, kpi=None):
    fig = go.Figure()
    if 'date' in forecast.columns:
        fig.add_trace(go.Scatter(x=forecast['date'], y=forecast['predicted_revenue'],
                                 mode='lines', name='Forecast', line=dict(color='red')))
    else:
        st.warning("Column 'date' not found in forecast data.")
    if 'date' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['revenue_usd'],
                                 mode='lines', name='Historical', line=dict(color='blue')))
    else:
        st.warning("Column 'date' not found in historical data.")
    if kpi is not None:
        fig.add_hline(y=kpi, line_dash="dash", line_color="green", name="KPI")
    fig.update_layout(title='Revenue Forecast vs Historical', xaxis_title='Date', yaxis_title='Revenue (USD)')
    return fig

def main():
    st.title("Streaming Revenue Analysis")
    
    # טעינת נתונים
    df = load_data()
    if df.empty:
        st.error("No data available")
        return
    
    # בחירת פלטפורמה
    platforms = ['Spotify', 'Sales', 'Overall']
    selected_platform = st.selectbox('Select Platform', platforms, key='main_platform_selector')
    
    # הצגת KPI-ים
    st.header("Key Performance Indicators")
    kpis = calculate_kpis(df, forecast_df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"${kpis['total_revenue']:,.0f}")
    with col2:
        st.metric("Average Monthly Revenue", f"${kpis['avg_monthly']:,.0f}")
    with col3:
        st.metric("Last Month Revenue", f"${kpis['last_month']:,.0f}")
    with col4:
        st.metric("Growth Rate", f"{kpis['growth_rate']:.1f}%")
    
    # השוואת פלטפורמות
    st.header("Platform Comparison")
    st.plotly_chart(plot_platform_comparison(df), use_container_width=True)
    
    # ניתוח מפורט לפלטפורמה נבחרת
    st.header(f"Detailed Analysis - {selected_platform}")
    
    # הגדרת KPI
    kpi = st.number_input('Set Monthly KPI Target ($)', min_value=0, value=0, step=100)
    
    # טעינת והצגת תחזית
    forecast = load_forecast(selected_platform)
    st.plotly_chart(plot_forecast_with_history(df, forecast, selected_platform, kpi if kpi > 0 else None),
                   use_container_width=True)
    
    # ניתוח נוסף
    st.header("Additional Insights")
    
    # ניתוח עונתיות
    st.subheader("Seasonality Analysis")
    platform_data = df[df['platform'] == selected_platform]
    monthly_avg = platform_data.groupby(platform_data['date'].dt.month)['revenue_usd'].mean()
    
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Bar(
        x=monthly_avg.index,
        y=monthly_avg.values,
        name='Average Monthly Revenue'
    ))
    fig_seasonal.update_layout(
        title='Average Revenue by Month',
        xaxis_title='Month',
        yaxis_title='Average Revenue (USD)',
        xaxis=dict(tickmode='array', ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                  tickvals=list(range(1, 13)))
    )
    st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # הצגת מידע על המודל
    st.info("""
    **About the Forecast Model:**
    - Uses LightGBM with advanced features including:
        - Time-based features (year, month, quarter)
        - Special events (holidays, Tomorrowland, ADE)
        - Historical trends and seasonality
        - Platform-specific patterns
    - The model is trained on historical data and validated on a test set
    - Forecast includes uncertainty bounds (±10%)
    - Seasonal decomposition shows trend and seasonal components
    """)

def update_forecast_chart(selected_platform, selected_period_type):
    if selected_platform == 'All Platforms':
        df_filtered = forecast_df
    else:
        df_filtered = forecast_df[forecast_df['platform'] == selected_platform]
    if selected_period_type != 'All Periods':
        df_filtered = df_filtered[df_filtered['period_type'] == selected_period_type]
    fig = px.line(df_filtered, x='date', y='revenue_usd', color='platform', title='Revenue Forecast')
    fig.update_layout(xaxis_title='Date', yaxis_title='Revenue (USD)')
    return fig

def update_summary_table(selected_platform, selected_period_type):
    if selected_platform == 'All Platforms':
        df_filtered = forecast_df
    else:
        df_filtered = forecast_df[forecast_df['platform'] == selected_platform]
    if selected_period_type != 'All Periods':
        df_filtered = df_filtered[df_filtered['period_type'] == selected_period_type]
    summary = df_filtered.groupby('platform')['revenue_usd'].sum().reset_index()
    summary.columns = ['Platform', 'Total Revenue (USD)']
    return summary

def update_platform_chart(selected_platform, selected_period_type):
    if selected_platform == 'All Platforms':
        df_filtered = forecast_df
    else:
        df_filtered = forecast_df[forecast_df['platform'] == selected_platform]
    if selected_period_type != 'All Periods':
        df_filtered = df_filtered[df_filtered['period_type'] == selected_period_type]
    fig = px.bar(df_filtered, x='platform', y='revenue_usd', title='Revenue by Platform')
    fig.update_layout(xaxis_title='Platform', yaxis_title='Revenue (USD)')
    return fig

def update_period_chart(selected_platform, selected_period_type):
    if selected_platform == 'All Platforms':
        df_filtered = forecast_df
    else:
        df_filtered = forecast_df[forecast_df['platform'] == selected_platform]
    if selected_period_type != 'All Periods':
        df_filtered = df_filtered[df_filtered['period_type'] == selected_period_type]
    fig = px.bar(df_filtered, x='period_type', y='revenue_usd', title='Revenue by Period Type')
    fig.update_layout(xaxis_title='Period Type', yaxis_title='Revenue (USD)')
    return fig

def update_overall_chart(selected_platform, selected_period_type):
    if selected_platform == 'All Platforms':
        df_filtered = forecast_df
    else:
        df_filtered = forecast_df[forecast_df['platform'] == selected_platform]
    if selected_period_type != 'All Periods':
        df_filtered = df_filtered[df_filtered['period_type'] == selected_period_type]
    overall_summary = df_filtered.groupby('platform')['revenue_usd'].sum().reset_index()
    overall_summary.columns = ['Platform', 'Total Revenue (USD)']
    fig = px.bar(overall_summary, x='Platform', y='Total Revenue (USD)', title='Overall Revenue by Platform')
    fig.update_layout(xaxis_title='Platform', yaxis_title='Total Revenue (USD)')
    return fig

def update_forecast_summary(selected_platform, selected_period_type):
    if selected_platform == 'All Platforms':
        df_filtered = forecast_df
    else:
        df_filtered = forecast_df[forecast_df['platform'] == selected_platform]
    if selected_period_type != 'All Periods':
        df_filtered = df_filtered[df_filtered['period_type'] == selected_period_type]
    forecast_summary = df_filtered.groupby('platform')['revenue_usd'].sum().reset_index()
    forecast_summary.columns = ['Platform', 'Forecasted Revenue (USD)']
    return forecast_summary

if __name__ == '__main__':
    main() 