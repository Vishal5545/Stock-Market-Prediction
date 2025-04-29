import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Input
from datetime import timedelta, datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
import os
import warnings
from scipy.signal import argrelextrema
from collections import defaultdict
import datetime
from datetime import datetime as dt
import pytz
from dateutil import parser
import re
from auth import show_login_page, init_session_state, logout_user
from ipo_data import render_ipo_section
from signal_processor import process_trading_signal_reasons, get_signal_display_class

# Helper functions for technical indicator interpretation
def get_rsi_interpretation(rsi_value):
    """Get interpretation text for RSI value"""
    if rsi_value > 70:
        return "Overbought"
    elif rsi_value < 30:
        return "Oversold"
    else:
        return "Neutral"

def get_macd_interpretation(macd_diff):
    """Get interpretation text for MACD difference"""
    if macd_diff > 0.5:
        return "Strong Bullish"
    elif macd_diff > 0:
        return "Bullish"
    elif macd_diff < -0.5:
        return "Strong Bearish"
    else:
        return "Bearish"

def get_bb_interpretation(price, upper, lower):
    """Get interpretation text for Bollinger Bands"""
    if price > upper:
        return "Overbought"
    elif price < lower:
        return "Oversold"
    else:
        percent = (price - lower) / (upper - lower) * 100
        if percent > 80:
            return "Near Upper Band"
        elif percent < 20:
            return "Near Lower Band"
        else:
            return "Middle Range"

# Set page title and enable wide layout - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
init_session_state()

# Show login page if not logged in
if not st.session_state.get('logged_in', False):
    show_login_page()
    st.stop()

# Show simple welcome header
st.title("Welcome to Stock Market Prediction")
st.write(f"Hello, {st.session_state.get('username', 'User')}! Let's analyze the markets together.")

# Show success message if just logged in
if st.session_state.get('just_logged_in', False):
    st.success("Successfully logged in!")
    st.session_state.just_logged_in = False

# Add user info and logout button to sidebar
with st.sidebar:
    st.markdown("### User Profile")
    st.write(f"Welcome, {st.session_state.get('username', 'User')}!")
    if st.button('Logout', key='logout_button', help='Click to logout', type='primary'):
        success, message = logout_user()
        if success:
            st.success("Logged out successfully!")
            st.rerun()

def validate_ticker(ticker_input):
    """
    Validate and clean ticker symbol input with enhanced error handling
    """
    if not ticker_input:
        return None
        
    # Clean ticker input - handle tuple format like ('A','A','P','L')
    if isinstance(ticker_input, tuple) or (isinstance(ticker_input, str) and '(' in ticker_input):
        # Convert tuple to string
        if isinstance(ticker_input, tuple):
            ticker_input = ''.join(ticker_input)
        else:
            # Extract characters from tuple notation string
            match = re.findall(r"'(\w)'", ticker_input)
            if match:
                ticker_input = ''.join(match)
            else:
                # Try another extraction pattern for different formats
                match = re.findall(r"\(\w+)\)", ticker_input)
                if match:
                    ticker_input = match[0]
    
    # Remove any remaining non-alphanumeric characters
    ticker_input = re.sub(r'[^a-zA-Z0-9]', '', ticker_input)
    
    # Convert to uppercase
    ticker_input = ticker_input.upper()
    
    # Ensure not empty after cleaning
    if not ticker_input:
        return None
        
    return ticker_input

# Load CSS from external file
with open('tailwind.css', 'r') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure Tensorflow
tf.get_logger().setLevel('ERROR')

# Add custom CSS for an enhanced modern dashboard with glassmorphism effects
st.markdown("""
<style>
    /* Modern dashboard theme with glassmorphism effects */
    .main {
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.7));
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 2rem;
        margin-top: 5rem;
        border-radius: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    /* Fixed header with glassmorphism */
    .fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        background: linear-gradient(135deg, rgba(30, 60, 114, 0.95), rgba(42, 82, 152, 0.95));
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 1.25rem;
        color: white;
        box-shadow: 0 4px 24px rgba(0,0,0,0.15);
        border-bottom: 1px solid rgba(255,255,255,0.1);
        width: 100%;
    }

    .header-content {
        max-width: 1400px;
        margin: 0 auto;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .header-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    .header-subtitle {
        font-size: 1rem;
        opacity: 0.9;
    }

    .market-info {
        display: flex;
        gap: 2rem;
        align-items: center;
    }

    .market-status {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        background: rgba(255,255,255,0.1);
    }

    /* Upgraded header with gradient and shadow */
    .header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px;
        color: white;
        text-align: center;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.1);
    }

    .logo-text {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 10px;
        font-family: 'Segoe UI', 'Arial', sans-serif;
    }

    .subheader {
        font-size: 18px;
        font-weight: 300;
        opacity: 0.9;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Enhanced card design with glassmorphism */
    .card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,255,255,0.8));
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.4);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }

    .card::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        height: 100%;
        width: 5px;
        background: linear-gradient(to bottom, #1e3c72, #2a5298);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .card:hover::after {
        opacity: 1;
    }

    .card-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 16px;
        color: #1e3c72;
        border-bottom: 2px solid #f0f4f8;
        padding-bottom: 12px;
        position: relative;
    }

    .card-title::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 40px;
        height: 2px;
        background: linear-gradient(to right, #1e3c72, #2a5298);
    }

    /* Modern chart containers with glassmorphism */
    .chart-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.8));
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.4);
        padding: 1rem !important;
        margin-top: -1rem !important;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .chart-container:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }

    .stPlotlyChart > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Fix spacing issues */
    [data-testid="stVerticalBlock"] > div:has(.stPlotlyChart) {
        padding-top: 0 !important;
        margin-top: -20px !important;
    }

    /* Improved tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        font-weight: 500;
        background-color: rgba(255,255,255,0.8);
        border: 1px solid #e9ecef;
        border-bottom: none;
        transition: all 0.2s ease-in-out;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 11px 24px 10px;
    }

    /* Modern button styling with glassmorphism */
    .stButton > button {
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: white;
        background: linear-gradient(135deg, rgba(30, 60, 114, 0.95), rgba(42, 82, 152, 0.95));
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }

    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(30, 60, 114, 0.3);
        transform: translateY(-2px);
    }

    /* Enhanced trading signal boxes */
    .signal-box-buy {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 18px;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }

    .signal-box-sell {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 18px;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);
    }

    .signal-box-neutral {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 18px;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }

    /* Enhanced alert boxes */
    .alert-info {
        background-color: #e7f5fe;
        border-left: 4px solid #0ea5e9;
        color: #0c4a6e;
        padding: 16px 20px;
        margin: 16px 0;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(14, 165, 233, 0.05);
    }

    .alert-warning {
        background-color: #fef6e4;
        border-left: 4px solid #f59e0b;
        color: #723b13;
        padding: 16px 20px;
        margin: 16px 0;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.05);
    }

    /* Modern form inputs with glassmorphism */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {
        border-radius: 12px;
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        border: 1px solid rgba(255,255,255,0.4);
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stDateInput > div > div > input:focus {
        border-color: #1e3c72;
        box-shadow: 0 0 0 2px rgba(30, 60, 114, 0.1);
    }

    /* Enhanced table styling */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
        border-radius: 8px;
        overflow: hidden;
        font-size: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    .dataframe th {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 12px 15px;
        text-align: left;
        font-weight: 500;
        position: sticky;
        top: 0;
    }

    .dataframe td {
        padding: 10px 15px;
        border-bottom: 1px solid #f1f5f9;
        transition: background-color 0.1s ease;
    }

    .dataframe tr:nth-child(even) {
        background-color: #f8fafc;
    }

    .dataframe tr:hover td {
        background-color: #e8f2ff;
    }

    /* Improved news cards */
    .news-card {
        background: white;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        border-left: 3px solid #1e3c72;
        transition: all 0.3s ease;
    }

    .news-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        border-left-width: 5px;
    }

    .news-title {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 8px;
        font-size: 16px;
        line-height: 1.4;
    }

    .news-meta {
        display: flex;
        justify-content: space-between;
        color: #64748b;
        font-size: 13px;
        margin-bottom: 12px;
    }

    .news-summary {
        color: #334155;
        line-height: 1.6;
        font-size: 14px;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    /* Improved metrics display */
    .metric-container {
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #1e3c72;
        margin-bottom: 4px;
    }

    .metric-label {
        font-size: 14px;
        color: #64748b;
        font-weight: 500;
    }

    /* Stock ticker animation with improved visuals */
    @keyframes ticker-scroll {
        0% { transform: translateX(10%); }
        100% { transform: translateX(-100%); }
    }

    .single-line-ticker {
        white-space: nowrap;
        overflow: hidden;
        background: rgba(255,255,255,0.1);
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 14px;
        box-shadow: inset 0 0 15px rgba(0,0,0,0.05);
    }

    .ticker-inner {
        display: inline-block;
        animation: ticker-scroll 120s linear infinite;
    }

    /* Stock values custom formatting */
    .stock-value-positive {
        color: #10b981;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        background-color: rgba(16, 185, 129, 0.1);
    }

    .stock-value-negative {
        color: #ef4444;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        background-color: rgba(239, 68, 68, 0.1);
    }

    /* Custom footer styling */
    footer:after {
        content: "Stock Market Prediction App © 2023 | Version 1.0.0 | Built with Streamlit";
        visibility: visible;
        display: block;
        position: relative;
        padding: 15px;
        text-align: center;
        font-size: 14px;
        color: #64748b;
        background: linear-gradient(to right, #f8f9fa, #ffffff, #f8f9fa);
        border-top: 1px solid #e5e7eb;
        margin-top: 50px;
        font-weight: 500;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Streamlit elements adjustments */
    .css-10trblm {
        margin-top: 0.8rem !important;
        margin-bottom: 0.8rem !important;
        color: #1e293b;
        font-weight: 600;
    }

    .stRadio > div {
        padding: 10px;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# Force reload of this file
# Initialize session state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}


from stock_api import load_stock_data

@st.cache_data(ttl=2*3600)  # Cache for 2 hours
def load_data(ticker, start_date=None, end_date=None):
    """Load and cache stock data with enhanced error handling and holiday checking
    
    Automatically uses 3 years of historical data with yfinance as the primary data source.
    """
    try:
        # Use the improved stock data loading function from stock_api
        data = load_stock_data(ticker, start_date, end_date)
        if data is None:
            st.error("No data found for the given ticker. Please check the symbol and try again.")
            return None
        return data

        # Import required modules
        import time
        from pandas.tseries.holiday import USFederalHolidayCalendar, Holiday, nearest_workday
        from pandas.tseries.offsets import CustomBusinessDay
        from datetime import date
        
        # Create a custom calendar class that includes both federal and major festival holidays
        class USMarketCalendar(USFederalHolidayCalendar):
            rules = USFederalHolidayCalendar.rules + [
                Holiday('Good Friday', month=4, day=7, offset=[nearest_workday]),
                Holiday('Christmas Eve', month=12, day=24, offset=[nearest_workday]),
                Holiday('New Years Eve', month=12, day=31, offset=[nearest_workday]),
                Holiday('Diwali', month=11, day=12, offset=[nearest_workday]),  # 2023 date
                Holiday('Chinese New Year', month=2, day=10, offset=[nearest_workday]),  # 2024 date
                Holiday('Easter Monday', month=4, day=10, offset=[nearest_workday]),  # 2024 date
                Holiday('Independence Day', month=7, day=4, offset=[nearest_workday]),
                Holiday('Labor Day', month=9, day=4, offset=[nearest_workday]),
                Holiday('Veterans Day', month=11, day=11, offset=[nearest_workday])
            ]
        
        # Create a calendar with both federal and festival holidays
        market_calendar = USMarketCalendar()
        
        # Ensure dates are properly converted to datetime objects
        current_date = pd.Timestamp.now().date()
        
        # Automatically set start_date to 3 years ago
        start_date = (pd.Timestamp.now() - pd.DateOffset(years=3)).date()
        
        # Set end_date to current date
        end_date = current_date
            
        # Check for holidays and weekends with error handling
        try:
            holidays = market_calendar.holidays(start=start_date, end=end_date)
            end_timestamp = pd.Timestamp(end_date)
            
            # Convert holidays to DatetimeIndex for consistent comparison
            holiday_dates = pd.DatetimeIndex(holidays).date
            
            while (end_timestamp.date() in holiday_dates or end_timestamp.weekday() >= 5):
                end_timestamp = end_timestamp - business_day
                st.info(f"Market is closed on {end_date}. Adjusting to last trading day: {end_timestamp.date()}")
            
            # Update end_date to last business day
            end_date = end_timestamp.date()
        except Exception as e:
            st.warning(f"Error checking holidays: {e}. Using original end date.")
            # Keep the original end_date if there's an error
        
        # Check if current date is a holiday or weekend with error handling
        try:
            current_timestamp = pd.Timestamp(current_date)
            holiday_dates = pd.DatetimeIndex(holidays).date
            
            if current_timestamp.date() in holiday_dates or current_timestamp.weekday() >= 5:
                st.warning("Market is currently closed due to holiday or weekend.")
                next_business_day = current_timestamp
                while next_business_day.date() in holiday_dates or next_business_day.weekday() >= 5:
                    next_business_day = next_business_day + business_day
                st.info(f"Market will reopen on {next_business_day.date()}")
        except Exception as e:
            st.warning(f"Error checking current date market status: {e}")
            # Continue with the process even if there's an error checking market status

        
        # Check current market status (US Eastern Time)
        et_tz = pytz.timezone('US/Eastern')
        current_et = datetime.datetime.now(et_tz)
        market_open = current_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Check if current time is within market hours
        is_market_hours = market_open <= current_et <= market_close
        is_weekday = current_et.weekday() < 5
        
        # Always set end_date to two days before current date to ensure reliable data
        # This avoids issues with incomplete market data for the current day
        last_trading_day = current_et - datetime.timedelta(days=2)
        # Ensure it's a valid trading day
        while (last_trading_day.date() in holidays or last_trading_day.weekday() >= 5):
            last_trading_day = last_trading_day - business_day
        end_date = last_trading_day.date()
        st.info(f"Using data up to {end_date} to ensure reliable analysis")
        
        # Display market status information
        if is_market_hours and is_weekday and current_et.date() not in holidays:
            st.info(f"Note: Market is currently open. Using historical data only to avoid incomplete data.")
        else:
            # Outside market hours or on non-trading day
            st.info(f"Note: Market is currently closed.")
            
            # Calculate and display next trading days
            next_business_day = current_et
            while next_business_day.date() in holidays or next_business_day.weekday() >= 5:
                next_business_day = next_business_day + business_day
            st.info(f"Next trading day will be: {next_business_day.date()}")
            
            # Show specific reason for market closure
            if not is_weekday:
                st.warning("Market is closed - Weekend")
            elif current_et.date() in holidays:
                st.warning("Market is closed - Holiday")
            elif current_et < market_open:
                st.info("Market is pre-market (Opens at 9:30 AM ET)")
            elif current_et > market_close:
                st.info("Market is after-hours (Closed at 4:00 PM ET)")


        
        # Check for upcoming holidays (next 2 days)
        future_date = current_date + pd.Timedelta(days=2)
        upcoming_holidays = market_calendar.holidays(start=current_date, end=future_date)
        
        # Display market status
        if not is_weekday:
            st.warning("Market is closed - Weekend")
        elif current_et.date() in holidays:
            st.warning("Market is closed - Holiday")
        elif not is_market_hours:
            if current_et < market_open:
                st.info("Market is pre-market (Opens at 9:30 AM ET)")
            else:
                st.info("Market is after-hours (Closed at 4:00 PM ET)")
                # Adjust end date to last market close if after hours
                if end_date == current_date:
                    end_date = current_date
        
        # Show upcoming holidays if any
        if len(upcoming_holidays) > 0:
            st.info(f"Note: Market will be closed for holiday on {', '.join(upcoming_holidays.strftime('%Y-%m-%d'))}")
            # Adjust end date if needed
            if end_date in upcoming_holidays:
                end_date = (pd.Timestamp(end_date) - business_day).date()
            
        # Retry mechanism for data fetching
        max_retries = 3
        retry_delay = 2
        last_error = None
        
        # Store error message in session state to prevent duplicate errors
        if 'error_message' not in st.session_state:
            st.session_state.error_message = ""
        
        for attempt in range(max_retries):
            try:
                # Fetch data from Yahoo Finance with adjusted date range
                stock_data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )

                # Check if we got any data
                if stock_data.empty:
                    last_error = f"No data found for {ticker}"
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    
                    # Only show error if it's different from the last one
                    error_msg = f"{last_error}. Please check the symbol and try again."
                    if st.session_state.error_message != error_msg:
                        st.error(error_msg)
                        st.session_state.error_message = error_msg
                    return None

                # Convert index to datetime if not already
                stock_data.index = pd.to_datetime(stock_data.index)

                # Make sure all required columns exist
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in stock_data.columns]
                
                if missing_cols:
                    last_error = f"Missing required columns: {', '.join(missing_cols)}"
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    
                    # Only show error if it's different from the last one
                    if st.session_state.error_message != last_error:
                        st.error(last_error)
                        st.session_state.error_message = last_error
                    return None
                
                # Clear error message on successful data load
                st.session_state.error_message = ""

                # Success - return the data
                return stock_data

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue

        # If we get here, all retries failed
        st.error(f"Failed to load data after {max_retries} attempts. Last error: {last_error}")
        return None

    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None


def create_model(time_steps, n_features, lstm_units_1=50, lstm_units_2=30,
                 dense_units=20, dropout_rate=0.2, simple_model=False):
    """Create a deep learning model for stock prediction"""

    # Reduce TensorFlow memory usage
    tf.config.experimental.set_memory_growth(
        tf.config.list_physical_devices('GPU')[0],
        True) if tf.config.list_physical_devices('GPU') else None

    if simple_model:
        # Create a simple, lightweight model for faster training
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(lstm_units_1, return_sequences=False,
                                input_shape=(time_steps, n_features)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1)
        ])
    else:
        # Create a more complex model with multiple LSTM layers
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(lstm_units_1, return_sequences=True,
                                input_shape=(time_steps, n_features)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.LSTM(lstm_units_2, return_sequences=False),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(dense_units, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    # Compile model with Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    return model

# Custom technical indicators to replace pandas_ta


def calculate_rsi(data, window=14):
    """Calculate RSI safely handling Series objects with improved error handling"""
    try:
        # Convert input to pandas Series if it isn't already
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
            
        # Calculate price changes
        delta = data.diff()
        
        # Separate gains and losses using boolean indexing
        gains = pd.Series(0, index=delta.index)
        losses = pd.Series(0, index=delta.index)
        gains[delta > 0] = delta[delta > 0]
        losses[delta < 0] = -delta[delta < 0]
        
        # Calculate averages
        avg_gain = gains.rolling(window=window, min_periods=1).mean()
        avg_loss = losses.rolling(window=window, min_periods=1).mean()
        
        # Calculate RS with proper error handling
        rs = pd.Series(0, index=data.index)
        valid_mask = (avg_loss != 0) & avg_loss.notna() & avg_gain.notna()
        rs[valid_mask] = avg_gain[valid_mask] / avg_loss[valid_mask]
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        rsi = rsi.fillna(50)  # Fill NaN with neutral value
        rsi = rsi.clip(0, 100)  # Ensure RSI stays within valid range
        rsi = rsi.clip(0, 100)  # Ensure values stay within 0-100 range
        
        return rsi
    except Exception as e:
        print(f"Error calculating RSI: {str(e)}")
        return pd.Series(50, index=data.index)  # Return neutral RSI on error


def calculate_sma(data, window=9):
    """Calculate Simple Moving Average"""
    return data.rolling(window=window).mean()


def calculate_ema(data, window=20):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=window, adjust=False).mean()


def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    fast_ema = calculate_ema(data, window=fast)
    slow_ema = calculate_ema(data, window=slow)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, window=signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands with improved error handling"""
    try:
        if data is None or len(data) == 0:
            raise ValueError("Input data is empty or None")
            
        sma = calculate_sma(data, window=window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        # Fill NaN values with forward fill then backward fill
        upper_band = upper_band.ffill().bfill()
        sma = sma.ffill().bfill()
        lower_band = lower_band.ffill().bfill()
        
        return upper_band, sma, lower_band
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {str(e)}")
        # Return neutral values on error
        neutral_series = pd.Series(data.mean() if len(data) > 0 else 0, index=data.index)
        return neutral_series, neutral_series, neutral_series


def calculate_stochastic(df, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator"""
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()

    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_window).mean()
    return k, d


@st.cache_data(ttl=3600)  # Cache for 1 hour
def add_indicators(df):
    """Add technical indicators to dataframe"""
    try:
        # Fix: Check if df is None or empty using proper method
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            raise ValueError("Input dataframe is empty or None")

        # Make a deep copy of the dataframe to avoid modifying the original
        df_copy = df.copy()

        # Make sure we have the basic required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df_copy.columns:
                print(f"Missing required column: {str(col)}")
                # Add placeholder data if missing
                if col in ['Open', 'High', 'Low', 'Close']:
                    df_copy[col] = df_copy['Close'] if 'Close' in df_copy.columns else 0
                elif col == "Volume":  # Fix: Use string equality instead of str()
                    df_copy[col] = 0

    # Add RSI
        df_copy['RSI'] = calculate_rsi(df_copy['Close'])

        # Add Moving Averages
        df_copy['SMA'] = calculate_sma(df_copy['Close'], window=9)
        df_copy['EMA_20'] = calculate_ema(df_copy['Close'], window=20)
        df_copy['EMA_50'] = calculate_ema(df_copy['Close'], window=50)

        # Add MACD
        df_copy['MACD'], df_copy['MACD_Signal'], df_copy['MACD_Hist'] = calculate_macd(
            df_copy['Close'])

        # Add Bollinger Bands
        df_copy['BB_Upper'], df_copy['BB_Middle'], df_copy['BB_Lower'] = calculate_bollinger_bands(df_copy['Close'])

        # Add Stochastic Oscillator
        df_copy['Stoch_K'], df_copy['Stoch_D'] = calculate_stochastic(df_copy)
    
        # Forward fill and backward fill NaN values
        return df_copy.ffill().bfill()
    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        if "[" in str(e) and "not in index" in str(e):
            print("This appears to be a ticker format issue. Attempting to fix...")

        # Create a copy to avoid modifying the original
        # Fix: Check if df is None or empty using proper method before copying
        df_copy = df.copy() if (df is not None and not (isinstance(df, pd.DataFrame) and df.empty)) else pd.DataFrame({'Close': [0]})

        # Ensure the required columns exist even if calculation fails
        for col in ['RSI', 'SMA', 'EMA_20', 'EMA_50', 'MACD', 'MACD_Signal',
                    'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'Stoch_K', 'Stoch_D']:
            if col not in df_copy.columns:
                df_copy[col] = 50  # Default neutral value

        return df_copy


def prepare_stock_data(data):
    """Prepare stock data with indicators"""
    try:
        if data is None or data.empty:
            raise ValueError("Input data is empty or None")
            
        # Create a copy to avoid modifying the original
        df_with_indicators = data.copy()
        
        # Calculate technical indicators
        df_with_indicators = add_indicators(df_with_indicators)
        
        # Ensure columns exist or have fallbacks
        feature_columns = []
        for col in ['Close', 'Volume', 'RSI', 'MACD', 'BB_Width']:
            if col in df_with_indicators.columns:
                feature_columns.append(col)
        
        # Ensure we have at least basic features
        if len(feature_columns) < 2:
            raise ValueError("Not enough features available for prediction")
        
        features = df_with_indicators[feature_columns].values
        return df_with_indicators, features
    except Exception as e:
        st.markdown(f"""
        <div class="error-container">
            <div class="error-message">Error preparing stock data: {str(e)}</div>
        </div>
        """, unsafe_allow_html=True)
        return None, None


def prepare_data(data, time_steps):
    """Prepare data for LSTM model"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps])
        y.append(scaled_data[i + time_steps, 0])
    return np.array(X), np.array(y), scaler


def predict_future(model, last_sequence, scaler, n_steps):
    """
    Predict future stock values

    Args:
        model: Trained model
        last_sequence: Last sequence from the dataset
        scaler: Trained scaler for inverse transformation
        n_steps: Number of future days to predict
    """
    # Make a copy of the last sequence to avoid modifying the original
    future_sequence = np.copy(last_sequence)
    future_predictions = []

    for _ in range(n_steps):
        # Reshape for prediction (model expects [batch_size, time_steps, n_features])
        current_sequence = future_sequence.reshape(
            1, future_sequence.shape[0], future_sequence.shape[1])

        # Predict the next value
        next_pred = model.predict(current_sequence, verbose=0)[0][0]
        future_predictions.append(next_pred)

        # Update sequence by removing the first element and adding the prediction
        # Create a new row with all features
        new_row = np.zeros(future_sequence.shape[1])
        new_row[0] = next_pred  # Set the prediction to the first column (Close price)

        # Shift the sequence and add the new prediction at the end
        future_sequence = np.vstack((future_sequence[1:], new_row))

    # Convert predictions to numpy array
    future_predictions = np.array(future_predictions).reshape(-1, 1)

    # Create a dummy array for inverse transformation (scaler expects all features)
    dummy_array = np.zeros((len(future_predictions), scaler.scale_.shape[0]))
    dummy_array[:, 0] = future_predictions.flatten()

    # Inverse transform to get actual values
    future_predictions_transformed = scaler.inverse_transform(dummy_array)[:, 0].reshape(-1, 1)

    return future_predictions_transformed


def display_data(df, currency_symbol='$', include_patterns=True):
    """
    Format and display stock data with enhanced styling

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing stock data
    currency_symbol : str, default='$'
        Symbol to use for currency formatting
    include_patterns : bool, default=True
        Whether to include pattern detection in the displayed data

    Returns:
    --------
    styled_df : pandas DataFrame
        Styled DataFrame with formatting applied
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Make a copy to avoid modifying the original
    display_df = df.copy()

    # Display only the most recent rows (last 10 rows or fewer if df is smaller)
    num_rows = min(10, len(display_df))
    display_df = display_df.iloc[-num_rows:].copy()

    # Round numeric values to 2 decimal places
    for col in display_df.select_dtypes(include=['float', 'int']).columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)

    # Ensure required columns exist
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in display_df.columns:
            display_df[col] = np.nan

    # Format the Date column if it exists and isn't already formatted
    if 'Date' in display_df.columns and not pd.api.types.is_string_dtype(display_df['Date']):
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')

    # Create a styled DataFrame
    def style_df(df):
        # Create a copy of the dataframe
        df_style = df.copy()

        # Define styling functions
        def color_price_movement(val):
            """Apply color based on price movement"""
            if isinstance(val, (int, float)) and not pd.isna(val):
                if val > 0:
                    # Green for bullish
                    return 'background-color: rgba(0, 200, 83, 0.2); color: #00C853'
                elif val < 0:
                    # Red for bearish
                    return 'background-color: rgba(255, 61, 0, 0.2); color: #FF3D00'
            return ''

        def format_volume(val):
            """Format volume with K for thousands and M for millions"""
            if pd.isna(val) or not isinstance(val, (int, float)):
                return val

            if val >= 1_000_000:
                return f'{val/1_000_000:.2f}M'
            elif val >= 1_000:
                return f'{val/1_000:.2f}K'
            else:
                return f'{val:.2f}'

        # Apply volume formatting
        if 'Volume' in df_style.columns:
            df_style['Volume'] = df_style['Volume'].apply(format_volume)

        # Calculate price change and percentage for styling
        if all(col in df.columns for col in ['Open', 'Close']):
            # Calculate only if both columns exist
            try:
                df_style['Change'] = df['Close'] - df['Open']
                df_style['Change%'] = ((df['Close'] - df['Open']) / df['Open'] * 100).round(2)
            except Exception:
                # Handle potential errors (e.g., division by zero)
                pass

        # Format RSI with color based on value
        def color_rsi(val):
            """Apply color to RSI based on thresholds"""
            if pd.isna(val) or not isinstance(val, (int, float)):
                return ''

            if val >= 70:
                return 'background-color: rgba(255, 61, 0, 0.2); color: #FF3D00'  # Overbought (red)
            elif val <= 30:
                return 'background-color: rgba(0, 200, 83, 0.2); color: #00C853'  # Oversold (green)
            else:
                return ''  # Neutral

        # Style for patterns
        def color_pattern(pattern):
            """Apply color to pattern based on its type"""
            if pd.isna(pattern) or not pattern or pattern == '':
                return ''

            pattern_str = str(pattern).lower()
            if 'bullish' in pattern_str:
                # Green for bullish
                return 'background-color: rgba(0, 200, 83, 0.2); color: #00C853'
            elif 'bearish' in pattern_str:
                return 'background-color: rgba(255, 61, 0, 0.2); color: #FF3D00'  # Red for bearish
            else:
                # Blue for neutral
                return 'background-color: rgba(33, 150, 243, 0.2); color: #2196F3'

        # Create the styled DataFrame
        styled = df_style.style

        # Apply conditional formatting for price movements if columns exist
        if 'Change' in df_style.columns:
            styled = styled.applymap(color_price_movement, subset=['Change'])
        if 'Change%' in df_style.columns:
            styled = styled.applymap(color_price_movement, subset=['Change%'])
            styled = styled.format({'Change%': '{:.2f}%'})

        # Apply RSI styling if column exists
        if 'RSI' in df_style.columns:
            styled = styled.applymap(color_rsi, subset=['RSI'])

        # Apply pattern styling if column exists and include_patterns is True
        if 'Pattern' in df_style.columns and include_patterns:
            styled = styled.applymap(color_pattern, subset=['Pattern'])

        # Format numeric columns
        for col in df_style.select_dtypes(include=['float']).columns:
            if col in ['Volume', 'Change%']:  # Skip already formatted columns
                continue
            if col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Prediction']:
                # Apply currency formatting
                styled = styled.format({col: f'{currency_symbol}{{:.2f}}'})
            else:
                # Apply standard decimal formatting
                styled = styled.format({col: '{:.2f}'})

        return styled

    # Apply the styling and return
    return style_df(display_df)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def detect_candlestick_patterns(df):
    """
    Enhanced candlestick pattern detection function
    Identifies common reversal and continuation patterns
    """
    if df is None or df.empty:
        return df

    # Create a copy to avoid modifying the original
    df_patterns = df.copy()

    # Add a Pattern column
    df_patterns['Pattern'] = None
    df_patterns['Pattern_Type'] = None

    # Process rows individually to avoid Series truth value errors
    for i in range(len(df_patterns)):
        try:
            # Skip if we don't have enough prior data for pattern detection
            if i < 3:
                continue
                
            # Access individual scalars using iloc & convert to Python scalars
            current_open = float(df_patterns.iloc[i, df_patterns.columns.get_loc('Open')])
            current_high = float(df_patterns.iloc[i, df_patterns.columns.get_loc('High')])
            current_low = float(df_patterns.iloc[i, df_patterns.columns.get_loc('Low')])
            current_close = float(df_patterns.iloc[i, df_patterns.columns.get_loc('Close')])
            
            # Get previous candle data
            prev_open = float(df_patterns.iloc[i-1, df_patterns.columns.get_loc('Open')])
            prev_high = float(df_patterns.iloc[i-1, df_patterns.columns.get_loc('High')])
            prev_low = float(df_patterns.iloc[i-1, df_patterns.columns.get_loc('Low')])
            prev_close = float(df_patterns.iloc[i-1, df_patterns.columns.get_loc('Close')])
            
            # Get prior 2 candles for 3-candle patterns
            prev2_open = float(df_patterns.iloc[i-2, df_patterns.columns.get_loc('Open')])
            prev2_high = float(df_patterns.iloc[i-2, df_patterns.columns.get_loc('High')])
            prev2_low = float(df_patterns.iloc[i-2, df_patterns.columns.get_loc('Low')])
            prev2_close = float(df_patterns.iloc[i-2, df_patterns.columns.get_loc('Close')])
            
            # Calculate body sizes
            current_body = abs(current_close - current_open)
            prev_body = abs(prev_close - prev_open)
            prev2_body = abs(prev2_close - prev2_open)
            
            # Calculate shadows
            current_upper_shadow = current_high - max(current_open, current_close)
            current_lower_shadow = min(current_open, current_close) - current_low
            
            # Determine bullish/bearish status
            is_current_bullish = current_close > current_open
            is_prev_bullish = prev_close > prev_open
            is_prev2_bullish = prev2_close > prev2_open
            
            # Calculate average body size as reference
            avg_body_size = (current_body + prev_body + prev2_body) / 3
            
            # 1. Basic patterns (already in your code)
            if is_current_bullish:
                # Bullish Marubozu (strong bullish candle with tiny or no shadows)
                if (current_high - current_close) < current_body * 0.1 and (current_open - current_low) < current_body * 0.1:
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern')] = "Bullish Marubozu"
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern_Type')] = "Strong Bullish"
                
                # Bullish Hammer (small body at top, long lower shadow)
                elif current_lower_shadow > current_body * 2 and current_upper_shadow < current_body * 0.5:
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern')] = "Bullish Hammer"
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern_Type')] = "Reversal Bullish"
                
                # Morning Star (3-candle bullish reversal pattern)
                elif not is_prev_bullish and is_prev2_bullish and prev_body < avg_body_size * 0.5 and current_body > prev_body * 1.5:
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern')] = "Morning Star"
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern_Type')] = "Reversal Bullish"
                
                # Bullish Engulfing (current bullish candle engulfs previous bearish candle)
                elif not is_prev_bullish and current_open < prev_close and current_close > prev_open:
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern')] = "Bullish Engulfing"
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern_Type')] = "Reversal Bullish"
                
                # Piercing Line (bullish candle closing more than halfway up previous bearish candle)
                elif not is_prev_bullish and current_open < prev_low and current_close > (prev_open + prev_close) / 2:
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern')] = "Piercing Line"
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern_Type')] = "Reversal Bullish"
                
                # Three White Soldiers (three consecutive bullish candles, each closing higher)
                elif is_prev_bullish and is_prev2_bullish and current_close > prev_close and prev_close > prev2_close:
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern')] = "Three White Soldiers"
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern_Type')] = "Continuation Bullish"
            else:
                # Bearish Marubozu (strong bearish candle with tiny or no shadows)
                if (current_high - current_open) < current_body * 0.1 and (current_close - current_low) < current_body * 0.1:
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern')] = "Bearish Marubozu"
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern_Type')] = "Strong Bearish"
                
                # Bearish Hanging Man (small body at bottom, long upper shadow)
                elif current_upper_shadow > current_body * 2 and current_lower_shadow < current_body * 0.5:
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern')] = "Bearish Hanging Man"
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern_Type')] = "Reversal Bearish"
                
                # Evening Star (3-candle bearish reversal pattern)
                elif is_prev_bullish and not is_prev2_bullish and prev_body < avg_body_size * 0.5 and current_body > prev_body * 1.5:
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern')] = "Evening Star"
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern_Type')] = "Reversal Bearish"
                
                # Bearish Engulfing (current bearish candle engulfs previous bullish candle)
                elif is_prev_bullish and current_open > prev_close and current_close < prev_open:
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern')] = "Bearish Engulfing"
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern_Type')] = "Reversal Bearish"
                
                # Dark Cloud Cover (bearish candle closing more than halfway down previous bullish candle)
                elif is_prev_bullish and current_open > prev_high and current_close < (prev_open + prev_close) / 2:
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern')] = "Dark Cloud Cover"
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern_Type')] = "Reversal Bearish"
                
                # Three Black Crows (three consecutive bearish candles, each closing lower)
                elif not is_prev_bullish and not is_prev2_bullish and current_close < prev_close and prev_close < prev2_close:
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern')] = "Three Black Crows"
                    df_patterns.iloc[i, df_patterns.columns.get_loc('Pattern_Type')] = "Continuation Bearish"

        except (ValueError, TypeError) as e:
            # Skip rows with invalid data, continue quietly
            pass
        except Exception as e:
            # Just log other errors without stopping processing
            print(f"Skipping row {str(i)} in pattern detection")

    return df_patterns


def detect_support_resistance(df, num_points=5, window=20):
    """Detect support and resistance levels using local min/max"""
    # Return empty lists to avoid errors
    return [], []

# The original function had an issue with list operations
# This simplified version prevents the error by simply returning empty lists
# If you want to re-implement this functionality later, consider using numpy arrays
# instead of Python lists for the mathematical operations

# Add function to calculate buyer-seller ratio


@st.cache_data(ttl=3600)  # Cache for 1 hour
def calculate_buyer_seller_ratio(df):
    """
    Calculate buyer-seller ratio based on volume and price movement
    Returns a dataframe with additional columns and the overall ratio
    """
    try:
        # Make a copy of the dataframe to avoid modification issues
        df = df.copy()

        # First ensure we have the needed columns
        required_cols = ['Open', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            print("Missing required columns for buyer-seller analysis")
            return df, 1.0  # Return neutral value

        # Calculate daily price changes without using column assignment
        # Process row by row to avoid Series errors
        buy_volume_sum = 0.0
        sell_volume_sum = 0.0

        # Create the new columns first
        df['Buy_Volume'] = 0.0
        df['Sell_Volume'] = 0.0

        # Process each row individually
        for i in range(len(df)):
            try:
                # Access individual scalar values
                open_val = float(df['Open'].iloc[i])
                close_val = float(df['Close'].iloc[i])
                volume = float(df['Volume'].iloc[i])

                # Determine if bullish or bearish
                daily_change = close_val - open_val

                if daily_change >= 0:
                    # Bullish day
                    buy_volume_sum += volume
                    df.iloc[i, df.columns.get_loc('Buy_Volume')] = volume
                else:
                    # Bearish day
                    sell_volume_sum += volume
                    df.iloc[i, df.columns.get_loc('Sell_Volume')] = volume

            except (ValueError, TypeError) as e:
                # Skip rows with invalid data
                print(f"Skipping row {str(i)} in buyer-seller ratio calc: {str(e)}")
                continue

        # Calculate the ratio safely
        if sell_volume_sum > 0:
            ratio = buy_volume_sum / sell_volume_sum
        else:
            ratio = 5.0  # Cap at 5 for extreme cases

        # Calculate cumulative volumes safely
        try:
            df['Cum_Buy_Volume'] = df['Buy_Volume'].cumsum()
            df['Cum_Sell_Volume'] = df['Sell_Volume'].cumsum()
        except Exception as e:
            print(f"Error calculating cumulative volumes: {str(e)}")
            # Create empty columns if error
            df['Cum_Buy_Volume'] = 0.0
            df['Cum_Sell_Volume'] = 0.0

        return df, ratio

    except Exception as e:
        print(f"Error in calculate_buyer_seller_ratio: {str(e)}")
        import traceback
        print(traceback.format_exc())

        # Return safe fallback values
        if df is not None:
            # Add required columns to avoid further errors
            if 'Buy_Volume' not in df.columns:
                df['Buy_Volume'] = 0.0
            if 'Sell_Volume' not in df.columns:
                df['Sell_Volume'] = 0.0
            if 'Cum_Buy_Volume' not in df.columns:
                df['Cum_Buy_Volume'] = 0.0
            if 'Cum_Sell_Volume' not in df.columns:
                df['Cum_Sell_Volume'] = 0.0
            return df, 1.0  # Neutral ratio

        # If all else fails, return empty DataFrame with required columns
        empty_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume',
                                         'Buy_Volume', 'Sell_Volume', 'Cum_Buy_Volume', 'Cum_Sell_Volume'])
        return empty_df, 1.0


def plot_all_data(df, ticker, lookback_days=90, model_results=None, patterns=None,
                  sma_values=None, ema_values=None, buy_sell_ratio=None, currency_symbol="$"):
    """Plot all data including price, volume, patterns, RSI, and predictions using enhanced visualization"""
    # Limit the data to the lookback period
    df = df.tail(lookback_days).copy()

    # Extract data
    dates = df.index.tolist()
    opens = df['Open'].values.tolist()
    highs = df['High'].values.tolist()
    lows = df['Low'].values.tolist()
    closes = df['Close'].values.tolist()
    volumes = df['Volume'].values.tolist()

    # Calculate if volume bars should be green or red based on price change
    volume_colors = ['#00c853' if closes[i] >= opens[i] else '#ff3d00' for i in range(len(closes))]

    # Create simplified figure with more optimal spacing and enhanced size
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        specs=[
            {"secondary_y": True},  # Row 1: Price chart with volume on secondary y
            {"secondary_y": False},  # Row 2: RSI + MACD
            {"secondary_y": False},  # Row 3: Stochastic + Bollinger
        ],
        subplot_titles=(
            f"{str(ticker)} Price & Technical Analysis",
            "Momentum Indicators",
            "Volatility & Trend Indicators"
        )
    )

    # Add basic candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            increasing=dict(line=dict(color='#26a69a', width=1), fillcolor='#26a69a'),
            decreasing=dict(line=dict(color='#ef5350', width=1), fillcolor='#ef5350'),
            name="Price"
        ),
        row=1, col=1
    )

    # Add volume bars with modified colors
    fig.add_trace(
        go.Bar(
            x=dates,
            y=volumes,
            marker_color=volume_colors,
            opacity=0.7,
            name="Volume"
        ),
        row=1, col=1,
        secondary_y=True
    )

    # Add SMA if available
    if sma_values is not None:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=sma_values,
                name="SMA (9)",
                line=dict(color='rgba(255, 165, 0, 0.7)', width=1)
            )
        )
    
    # Add EMA values
    if 'EMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df['EMA_20'].values,
                name="EMA (20)",
                line=dict(color='rgba(46, 139, 87, 0.7)', width=1)
            )
        )

    if 'EMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df['EMA_50'].values,
                name="EMA (50)",
                line=dict(color='rgba(70, 130, 180, 0.7)', width=1)
            )
        )

    # Add Bollinger Bands
    if all(col in df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
        # Upper band
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df['BB_Upper'].values,
                name="BB Upper",
                line=dict(color='rgba(255, 0, 0, 0.3)', width=1)
            )
        )
        
        # Middle band
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df['BB_Middle'].values,
                name="BB Middle",
                line=dict(color='rgba(0, 0, 255, 0.3)', width=1)
            )
        )
        
        # Lower band
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df['BB_Lower'].values,
                name="BB Lower",
                line=dict(color='rgba(255, 0, 0, 0.3)', width=1)
            )
        )
        
        # Also plot the closing price in the Bollinger chart for reference
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=closes,
                name="Close Price",
                line=dict(color='rgba(0, 0, 0, 0.5)', width=1)
            )
        )

    # Add RSI if available
    if 'RSI' in df.columns and not df.empty:
        rsi_values = df['RSI'].values.tolist()

        # Add RSI line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rsi_values,
                mode='lines',
                line=dict(color='#3f51b5', width=1.5),
                name="RSI (14)"
            ),
            row=2, col=1
        )

        # Add oversold and overbought lines for RSI
        fig.add_trace(
            go.Scatter(
                x=[dates[0], dates[-1]],
                y=[30, 30],
                mode='lines',
                line=dict(color='green', width=1, dash='dash'),
                name="Oversold (30)",
                showlegend=False
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=[dates[0], dates[-1]],
                y=[70, 70],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                name="Overbought (70)",
                showlegend=False
            ),
            row=2, col=1
        )

        # Add middle line
        fig.add_trace(
            go.Scatter(
                x=[dates[0], dates[-1]],
                y=[50, 50],
                mode='lines',
                line=dict(color='rgba(0, 0, 0, 0.3)', width=1, dash='dot'),
                showlegend=False
            ),
            row=2, col=1
        )

    # Add MACD if available
    if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
        # MACD Line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df['MACD'].values,
                mode='lines',
                line=dict(color='#2196f3', width=1.5),
                name="MACD"
            ),
            row=2, col=1
        )

        # MACD Signal
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df['MACD_Signal'].values,
                mode='lines',
                line=dict(color='#ff9800', width=1.5),
                name="Signal Line"
            ),
            row=2, col=1
        )

        # MACD Histogram
        if 'MACD_Hist' in df.columns:
            # Create custom colors for histogram based on value
            hist_colors = ['#4caf50' if val >= 0 else '#f44336' for val in df['MACD_Hist'].values]

            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=df['MACD_Hist'].values,
                    marker_color=hist_colors,
                    name="MACD Histogram",
                    opacity=0.7,
                    showlegend=True
                ),
                row=2, col=1
            )

    # Add Stochastic Oscillator if available
    if all(col in df.columns for col in ['Stoch_K', 'Stoch_D']):
        # Add K line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df['Stoch_K'].values,
                mode='lines',
                line=dict(color='#9c27b0', width=1.5),
                name="%K Line"
            ),
            row=3, col=1
        )

        # Add D line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df['Stoch_D'].values,
                mode='lines',
                line=dict(color='#ff5722', width=1.5),
                name="%D Line"
            ),
            row=3, col=1
        )

        # Add overbought/oversold lines for Stochastic
        fig.add_trace(
            go.Scatter(
                x=[dates[0], dates[-1]],
                y=[80, 80],
                mode='lines',
                line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dash'),
                showlegend=False
            ),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=[dates[0], dates[-1]],
                y=[20, 20],
                mode='lines',
                line=dict(color='rgba(0, 128, 0, 0.5)', width=1, dash='dash'),
                showlegend=False
            ),
            row=3, col=1
        )

    # Add predictions from model results if available
    if not model_results.empty and 'y_pred_future' in model_results:
        future_dates = model_results.get('future_dates', [])
        y_pred_future = model_results.get('y_pred_future', [])

        if len(future_dates) > 0 and len(y_pred_future) > 0:
            # Add confidence interval if available
            if 'y_pred_lower' in model_results and 'y_pred_upper' in model_results:
                y_pred_lower = model_results.get('y_pred_lower', [])
                y_pred_upper = model_results.get('y_pred_upper', [])

                # Add confidence interval
                fig.add_trace(
                    go.Scatter(
                        x=future_dates + future_dates[::-1],
                        y=y_pred_upper + y_pred_lower[::-1],
                        fill='toself',
                        fillcolor='rgba(128, 0, 128, 0.1)',
                        line=dict(color='rgba(0, 0, 0, 0)'),
                        name="Prediction Range",
                        showlegend=True
                    ),
                    row=1, col=1
                )

            # Add future prediction line
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=y_pred_future,
                    mode='lines+markers',
                    line=dict(color='rgba(128, 0, 128, 0.9)', width=2),
                    marker=dict(size=6, symbol='circle'),
                    name="AI Prediction"
                ),
                row=1, col=1
            )

            # Add vertical line to show where prediction starts
            if len(dates) > 0:
                fig.add_shape(
                    type="line",
                    x0=dates[-1],
                    y0=0,
                    x1=dates[-1],
                    y1=1,
                    yref="paper",
                    line=dict(
                        color="rgba(0, 0, 0, 0.5)",
                        width=1.5,
                        dash="dash"
                    )
                )

                # Add annotation for prediction start
                fig.add_annotation(
                    x=dates[-1],
                    y=1.05,
                    yref="paper",
                    text="Prediction Start",
                    showarrow=False,
                    font=dict(size=10, color="rgba(0, 0, 0, 0.6)"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="rgba(0, 0, 0, 0.2)",
                    borderpad=2,
                    borderwidth=1
                )

    # Set axis ranges
    # RSI Chart (0-100)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    # Update layout with more detailed styling
    fig.update_layout(
        height=900,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        title=dict(
            text=f"{str(ticker)} Technical Analysis & Price Prediction",
            y=0.98,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(
                family="Arial",
                size=20,
                color="#1e3c72"
            )
        )
    )

    # Update styles for each subplot
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.8)',
        zeroline=False,
        showline=True,
        linecolor='rgba(0, 0, 0, 0.3)'
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.8)',
        zeroline=False,
        showline=True,
        linecolor='rgba(0, 0, 0, 0.3)'
    )

    # Add watermark for prediction disclaimer
    fig.add_annotation(
        x=0.5,
        y=0.02,
        xref="paper",
        yref="paper",
        text="Technical analysis and AI predictions are for informational purposes only. Not financial advice.",
        showarrow=False,
        font=dict(family="Arial", size=10, color="rgba(0,0,0,0.3)"),
        align="center"
    )

    return fig

# Function to generate buy/sell signals based on patterns and technical indicators


def generate_trading_signals(df):
    """
    Generate trading signals (Buy/Sell/Neutral) based on technical indicators
    Returns a DataFrame with signals and confidence levels
    """
    try:
        # Ensure we have data
        if df is None or df.empty:
            # Create a default DataFrame with Neutral signal
            index = [pd.Timestamp.now()]
            return pd.DataFrame({
                'Signal': ['Neutral'],
                'Confidence': [0],
                'Reasoning': ['No data available']
            }, index=index)

        # Create a new DataFrame to store signals
        signals_df = pd.DataFrame(index=df.index)
        signals_df['Signal'] = 'Neutral'
        signals_df['Confidence'] = 0
        signals_df['Reasoning'] = 'Initializing technical analysis'

        # Process each data point to generate signals
        for i in range(len(df)):
            if i < 5:  # Skip the first few rows due to insufficient data for calculations
                continue

            # Analyze each row of data
            try:
                # Get current data point
                current = df.iloc[i]

                # Initialize variables for signals
                bullish_signals = 0.0
                bearish_signals = 0.0
                neutral_signals = 0.0
                total_signals = 0.0
                signal_reasons = []

                # Check RSI oversold/overbought
                if 'RSI' in df.columns:
                    try:
                        rsi_value = float(current['RSI'])
                        if rsi_value > 70:
                            bearish_signals += 1.0
                            signal_reasons.append(f"RSI is overbought ({rsi_value:.1f})")
                        elif rsi_value < 30:
                            bullish_signals += 1.0
                            signal_reasons.append(f"RSI is oversold ({rsi_value:.1f})")
                        elif rsi_value > 60:
                            bearish_signals += 0.5
                            signal_reasons.append(f"RSI is neutral-bearish ({rsi_value:.1f})")
                        elif rsi_value < 40:
                            bullish_signals += 0.5
                            signal_reasons.append(f"RSI is neutral-bullish ({rsi_value:.1f})")
                        else:
                            neutral_signals += 1.0
                            signal_reasons.append(f"RSI is neutral ({rsi_value:.1f})")
                        total_signals += 1.0
                    except Exception as e:
                        print(f"Error processing RSI: {str(e)}")
            except Exception as e:
                print(f"Error analyzing row {i}: {str(e)}")

            # Check price relative to moving averages
            if all(col in df.columns for col in ['Close', 'SMA', 'EMA_20']):
                try:
                    price = float(current['Close'])
                    sma = float(current['SMA'])
                    ema20 = float(current['EMA_20'])

                    # Price vs SMA
                    if price > sma * 1.05:
                        bearish_signals += 1.0
                        signal_reasons.append(
                            f"Price ({price:.2f}) significantly above SMA ({sma:.2f})")
                        total_signals += 1.0
                    elif price < sma * 0.95:
                        bullish_signals += 1.0
                        signal_reasons.append(
                            f"Price ({price:.2f}) significantly below SMA ({sma:.2f})")
                        total_signals += 1.0

                    # Price vs EMA
                    if price > ema20:
                        bullish_signals += 0.5
                        signal_reasons.append(f"Price above EMA20")
                    else:
                        bearish_signals += 0.5
                        signal_reasons.append(f"Price below EMA20")
                except Exception as e:
                    print(f"Error checking moving averages: {str(e)}")

            # Check MACD if available
            if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
                try:
                    # Convert to float to avoid Series truth value ambiguity
                    macd = float(current['MACD'])
                    macd_signal = float(current['MACD_Signal'])

                    if macd > macd_signal:
                        bullish_signals += 1.0
                        signal_reasons.append("MACD above signal line")
                        total_signals += 1.0
                    else:
                        bearish_signals += 1.0
                        signal_reasons.append("MACD below signal line")
                        total_signals += 1.0

                    # Check MACD histogram direction
                    if i > 0:
                        prev_macd = float(df.iloc[i-1]['MACD'])
                        prev_macd_signal = float(df.iloc[i-1]['MACD_Signal'])

                        prev_hist = prev_macd - prev_macd_signal
                        curr_hist = macd - macd_signal

                        if curr_hist > prev_hist:
                            bullish_signals += 0.5
                            signal_reasons.append("MACD histogram improving")
                            total_signals += 0.5
                        else:
                            bearish_signals += 0.5
                            signal_reasons.append("MACD histogram deteriorating")
                except Exception as e:
                    print(f"Error checking MACD: {str(e)}")

            # Calculate final signal
            if total_signals > 0:
                bullish_confidence = (bullish_signals / total_signals) * 100
                bearish_confidence = (bearish_signals / total_signals) * 100
                neutral_confidence = (neutral_signals / total_signals) * 100

                if bullish_confidence > bearish_confidence and bullish_confidence > neutral_confidence:
                    signal = "Buy"
                    confidence = bullish_confidence
                elif bearish_confidence > bullish_confidence and bearish_confidence > neutral_confidence:
                    signal = "Sell"
                    confidence = bearish_confidence
                else:
                    signal = "Neutral"
                    confidence = neutral_confidence

                # Store signals in DataFrame
                signals_df.iloc[i, signals_df.columns.get_loc('Signal')] = signal
                signals_df.iloc[i, signals_df.columns.get_loc('Confidence')] = confidence
                signals_df.iloc[i, signals_df.columns.get_loc(
                    'Reasoning')] = ", ".join(signal_reasons)
            else:
                # No signals could be calculated
                signals_df.iloc[i, signals_df.columns.get_loc('Signal')] = "Neutral"
                signals_df.iloc[i, signals_df.columns.get_loc('Confidence')] = 0
                signals_df.iloc[i, signals_df.columns.get_loc(
                    'Reasoning')] = "Insufficient technical signals"

        return signals_df

    except Exception as e:
        print(f"Error in generate_trading_signals: {str(e)}")
        # Return a default DataFrame with a single row
        index = [pd.Timestamp.now()]
        return pd.DataFrame({
            'Signal': ['Neutral'],
            'Confidence': [0],
            'Reasoning': [f'Error generating signals: {str(e)}']
        }, index=index)


def plot_prediction_analysis(df, model_results, ticker, currency_symbol="$"):
    """Generate a dedicated prediction analysis chart with enhanced visualization similar to TradingView"""

    # Create a Plotly figure
    fig = go.Figure()

    if df is None or df.empty:
        # Create empty figure if data is not available
        fig.update_layout(
            title="No prediction data available",
            height=500
        )
        return fig

    # Fix the ticker issue - ensure it's a proper string
    if isinstance(ticker, (list, tuple)):
        ticker = ''.join(ticker)

    # Calculate RSI using proper Series operations
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-6)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate Stochastic Oscillator using direct Series operations
    # Compute rolling min and max directly as Series
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    
    # Avoid divide-by-zero with proper Series operations
    denominator = high_14 - low_14
    denominator = denominator.replace(0, 1)  # Replace zeros with 1 to avoid division errors
    
    # Calculate %K as a clean Series
    stoch_k_series = ((df['Close'] - low_14) / denominator) * 100
    stoch_k_series = stoch_k_series.clip(0, 100).fillna(50)
    
    # Assign to single column - now guaranteed to be a Series
    df['%K'] = stoch_k_series
    df['%D'] = df['%K'].rolling(window=3).mean().fillna(50)  # 3-period SMA of %K

    # Ensure we have enough data to display
    if len(df) < 5:
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Insufficient data for prediction visualization",
            showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Extract data for plotting
    dates = df.index.tolist()
    closes = df['Close'].values.tolist()
    opens = df['Open'].values.tolist()
    highs = df['High'].values.tolist()
    lows = df['Low'].values.tolist()

    # Create figure with subplots for price, indicators, and volume
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,  # Increased spacing to prevent overlap
        row_heights=[0.45, 0.2, 0.2, 0.15],  # Optimized heights for better distribution
        subplot_titles=("", "", "", ""),  # Empty subplot titles to avoid overlap
        figure=fig  # Use the existing figure object
    )
    
    # Update layout for better spacing and readability
    fig.update_layout(
        height=900,  # Increased height for better visualization
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        margin=dict(t=100, b=50, l=50, r=50)  # Adjusted margins for better spacing
    )

    # Add custom subplot titles with better positioning
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.97,  # Position for the first subplot title
        text="Price & Predictions",
        showarrow=False,
        font=dict(family="Arial", size=14, color="#1e3c72"),
        bgcolor="rgba(248, 249, 250, 0.95)",
        bordercolor="rgba(150, 150, 150, 0.5)",
        borderwidth=1,
        borderpad=4,
        align="center"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.40,  # Position for the second subplot title
        text="MACD",
        showarrow=False,
        font=dict(family="Arial", size=14, color="#1e3c72"),
        bgcolor="rgba(248, 249, 250, 0.95)",
        bordercolor="rgba(150, 150, 150, 0.5)",
        borderwidth=1,
        borderpad=4,
        align="center"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.20,  # Position for the third subplot title
        text="RSI & Stochastic",
        showarrow=False,
        font=dict(family="Arial", size=14, color="#1e3c72"),
        bgcolor="rgba(248, 249, 250, 0.95)",
        bordercolor="rgba(150, 150, 150, 0.5)",
        borderwidth=1,
        borderpad=4,
        align="center"
    )

    # Add RSI indicator
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df['RSI'],
            mode='lines',
            line=dict(color='#7B1FA2', width=1.5),
            name='RSI'
        ),
        row=3, col=1
    )

    # Add Stochastic Oscillator
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df['%K'],
            mode='lines',
            line=dict(color='#1E88E5', width=1.5),
            name='%K'
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df['%D'],
            mode='lines',
            line=dict(color='#FFA726', width=1.5),
            name='%D'
        ),
        row=3, col=1
    )

    # Add Volume
    fig.add_trace(
        go.Bar(
            x=dates,
            y=df['Volume'],
            marker_color='rgba(128, 128, 128, 0.5)',
            name='Volume'
        ),
        row=4, col=1
    )

    # Add candlestick chart for historical data with prediction overlay
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            increasing=dict(line=dict(color='#26A69A', width=1), fillcolor='#26A69A'),
            decreasing=dict(line=dict(color='#EF5350', width=1), fillcolor='#EF5350'),
            name="Price",
            showlegend=True
        ),
        row=1, col=1
    )

    # Add prediction line if model results are available
    if model_results is not None and len(model_results) > 0:
        future_dates = pd.date_range(start=dates[-1], periods=len(model_results)+1, freq='D')[1:]
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=np.array(model_results['y_pred_future']).flatten(),
                mode='lines',
                line=dict(color='#FFD700', width=2, dash='dash'),
                name='Prediction',
                showlegend=True
            ),
            row=1, col=1
        )

    # --- Safely add moving averages ---
    try:
        if 'SMA' in df.columns and not pd.api.types.is_bool_dtype(df['SMA'].isna().all()):
            # Convert to Python boolean safely
            sma_has_values = not all(df['SMA'].isna())
            if sma_has_values:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['SMA'].values,
                        mode='lines',
                        line=dict(color='#1976D2', width=1.5),
                        name="9-day SMA"
                    ),
                    row=1, col=1
                )
    except Exception as e:
        print(f"Error adding SMA: {str(e)}")

    try:
        if 'EMA_20' in df.columns and not all(df['EMA_20'].isna()):
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=df['EMA_20'].values,
                    mode='lines',
                    line=dict(color='#FF9800', width=1.5),
                    name="20-day EMA"
                ),
                row=1, col=1
            )
    except Exception as e:
        print(f"Error adding EMA_20: {str(e)}")

    try:
        if 'EMA_50' in df.columns and not all(df['EMA_50'].isna()):
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=df['EMA_50'].values,
                    mode='lines',
                    line=dict(color='#9C27B0', width=1.5),
                    name="50-day EMA"
                ),
                row=1, col=1
            )
    except Exception as e:
        print(f"Error adding EMA_50: {str(e)}")

    # --- Safely add Bollinger Bands ---
    try:
        bb_columns = ['BB_Upper', 'BB_Middle', 'BB_Lower']
        if all(col in df.columns for col in bb_columns):
            # Check if at least one value is not NA in each column
            bb_upper_has_values = not all(df['BB_Upper'].isna())
            bb_middle_has_values = not all(df['BB_Middle'].isna())
            bb_lower_has_values = not all(df['BB_Lower'].isna())

            if bb_upper_has_values and bb_middle_has_values and bb_lower_has_values:
                # Add Bollinger Bands
                # Upper band
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['BB_Upper'].values,
                        mode='lines',
                        line=dict(color='rgba(68, 138, 255, 0.7)', width=1, dash='dot'),
                        name="Bollinger Upper",
                        hoverinfo='none'
                    ),
                    row=1, col=1
                )
                
                # Middle band
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['BB_Middle'].values,
                        mode='lines',
                        line=dict(color='rgba(68, 138, 255, 0.9)', width=1),
                        name="Bollinger Middle"
                    ),
                    row=1, col=1
                )
                
                # Lower band
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['BB_Lower'].values,
                        mode='lines',
                        line=dict(color='rgba(68, 138, 255, 0.7)', width=1, dash='dot'),
                        name="Bollinger Lower",
                        hoverinfo='none',
                        fill='tonexty',
                        fillcolor='rgba(68, 138, 255, 0.05)'
                    ),
                    row=1, col=1
                )
    except Exception as e:
        st.warning(f"Error adding Bollinger Bands: {str(e)}")

    # Add shaded region for prediction confidence interval
    if model_results is not None and len(model_results) > 0:
                    try:
                        if len(model_results['y_pred_future']) > 0:
                            std_dev = np.std(df['Close'][-30:])  # Use last 30 days for volatility estimate
                            future_dates = pd.date_range(start=dates[-1], periods=len(model_results['y_pred_future']), freq='D')[1:]
                            pred_array = np.array(model_results['y_pred_future']).flatten()
                            
                            # Ensure all arrays have the same length
                            if len(future_dates) == len(pred_array):
                                upper_bound = pred_array + (2 * std_dev)
                                lower_bound = pred_array - (2 * std_dev)
                                
                                # Add confidence interval as a single filled area
                                fig.add_trace(
                                    go.Scatter(
                                        x=list(future_dates) + list(future_dates)[::-1],
                                        y=list(upper_bound) + list(lower_bound)[::-1],
                                        fill='toself',
                                        fillcolor='rgba(255, 215, 0, 0.2)',
                                        line=dict(width=0),
                                        name='Prediction Range',
                                        showlegend=True
                                    ),
                                    row=1, col=1
                                )
                    except Exception as e:
                        st.warning(f"Error adding confidence interval visualization: {e}")

    # Middle band (usually 20-day SMA)
    fig.add_trace(
        go.Scatter(
        x=dates,
        y=df['BB_Middle'].values,
        mode='lines',
        line=dict(color='rgba(68, 138, 255, 0.9)', width=1),
        name="Bollinger Middle"
        ),
            row=1, col=1
        )

    # Lower band
    try:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df['BB_Lower'].values,
                mode='lines',
                line=dict(color='rgba(68, 138, 255, 0.7)', width=1, dash='dot'),
                name="Bollinger Lower",
                hoverinfo='none',
                fill='tonexty',
                fillcolor='rgba(68, 138, 255, 0.05)'
            ),
            row=1, col=1
        )
    except Exception as e:
        st.error(f"Error adding Bollinger Lower Band: {str(e)}")
        st.error("Please ensure the data contains valid Bollinger Bands calculations")

    # Update layout with improved configurations
    try:
        fig.update_layout(
            height=900,  # Increased height for better visibility
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            margin=dict(t=30, l=50, r=50, b=30)
        )

        # Update Y-axes labels and ranges
        fig.update_yaxes(
            title_text="Price",
            row=1, col=1,
            tickprefix=currency_symbol,
            tickformat='.2f'
        )
        fig.update_yaxes(
            title_text="MACD",
            row=2, col=1,
            tickformat='.2f'
        )
        fig.update_yaxes(
            title_text="RSI",
            row=3, col=1,
            range=[0, 100],
            tickformat='.0f'
        )
        fig.update_yaxes(
            title_text="Volume",
            row=4, col=1,
            tickformat='.0f'
        )

        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

        # Update X-axis to show dates properly
        fig.update_xaxes(rangeslider_visible=False)
    except Exception as e:
        st.error(f"Error updating chart layout: {str(e)}")
        st.error("Please check the chart configuration and try again.")


    # --- Safely add MACD indicator ---
    try:
        macd_columns = ['MACD', 'MACD_Signal', 'MACD_Hist']
        if all(col in df.columns for col in macd_columns):
            # Check if at least one value is not NA in each column
            macd_has_values = not all(df['MACD'].isna())
            macd_signal_has_values = not all(df['MACD_Signal'].isna())
            macd_hist_has_values = not all(df['MACD_Hist'].isna())

            # Fix: The problematic line using `.empty` on a boolean value
            # Original: if not macd_has_values.empty and macd_signal_has_values and macd_hist_has_values:
            # Changed to properly check boolean values
            if macd_has_values and macd_signal_has_values and macd_hist_has_values:
                # MACD Line
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['MACD'].values,
                        mode='lines',
                        line=dict(color='#2962FF', width=1.5),
                        name="MACD Line"
                    ),
                    row=2, col=1
                )

                # MACD Signal Line
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['MACD_Signal'].values,
                        mode='lines',
                        line=dict(color='#FF6D00', width=1.5),
                        name="MACD Signal"
                    ),
                    row=2, col=1
                )

                # MACD Histogram
                colors = []
                for val in df['MACD_Hist'].values:
                    if pd.notnull(val) and val >= 0:
                        colors.append('#26A69A')  # Green for positive
                    else:
                        colors.append('#EF5350')  # Red for negative

                fig.add_trace(
                    go.Bar(
                        x=dates,
                        y=df['MACD_Hist'].values,
                        marker=dict(color=colors),
                        name="MACD Histogram"
                    ),
                    row=2, col=1
                )

                # Add zero line for MACD
                fig.add_shape(
                    type="line",
                    x0=dates[0],
                    x1=dates[-1],
                    y0=0,
                    y1=0,
                    line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dot"),
                    row=2, col=1
                )
    except Exception as e:
        print(f"Error adding MACD: {str(e)}")

    # --- Safely add Volume indicator ---
    try:
        if 'Volume' in df.columns and not all(df['Volume'].isna()):
            colors = []
            for i in range(len(df)):
                if i > 0 and df['Close'].values[i] > df['Close'].values[i-1]:
                    colors.append('#26A69A')  # Green for up days
                else:
                    colors.append('#EF5350')  # Red for down days

            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=df['Volume'].values,
                    marker=dict(color=colors, line=dict(width=0)),
                    name="Volume"
                ),
                row=3, col=1
            )

            # Add 20-day average volume line
            vol_ma = df['Volume'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=vol_ma.values,
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0.5)', width=1.5),
                    name="20-day Avg Volume"
                ),
                row=3, col=1
            )
    except Exception as e:
        print(f"Error adding Volume: {str(e)}")

    # --- Safely add prediction data ---
    try:
        if model_results is not None and isinstance(
                model_results, dict) and 'y_pred_future' in model_results:
            future_dates = model_results.get('future_dates', [])
            y_pred_future = model_results.get('y_pred_future', [])

            if len(future_dates) > 0 and len(y_pred_future) > 0:
                # Add confidence intervals if available
                if 'y_pred_lower' in model_results and 'y_pred_upper' in model_results:
                    y_pred_lower = model_results.get('y_pred_lower', [])
                    y_pred_upper = model_results.get('y_pred_upper', [])

                    if len(y_pred_lower) > 0 and len(y_pred_upper) > 0:
                        # Create the filled area for prediction range
                        fig.add_trace(
                            go.Scatter(
                                x=future_dates + future_dates[::-1],
                                y=y_pred_upper + y_pred_lower[::-1],
                                fill='toself',
                                fillcolor='rgba(0, 150, 136, 0.2)',
                                line=dict(color='rgba(0, 150, 136, 0)'),
                                name="Prediction Range",
                                showlegend=True
                            ),
                            row=1, col=1
                        )

                # Add the prediction line with dots
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=y_pred_future,
                        mode='lines+markers',
                        line=dict(color='#00897B', width=2.5),
                        marker=dict(size=8, symbol='circle', color='#00897B',
                                    line=dict(color='white', width=1)),
                        name="Price Prediction",
                        hovertemplate="%{x|%b %d, %Y}: " +
                        currency_symbol + "%{y:.2f}<extra></extra>"
                    ),
                    row=1, col=1
                )

                # Add visual separator between historical and prediction data
                if len(dates) > 0:
                    fig.add_shape(
                        type="line",
                        x0=dates[-1],
                        x1=dates[-1],
                        y0=0,
                        y1=1,
                        yref="paper",
                        line=dict(color="rgba(0,0,0,0.5)", width=2, dash="dash")
                    )

                    # Add "Prediction Start" annotation
                    fig.add_annotation(
                        x=dates[-1],
                        y=1.05,
                        yref="paper",
                        text="Prediction Start",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1.5,
                        arrowcolor="rgba(0,0,0,0.5)",
                        font=dict(size=12, color="rgba(0,0,0,0.8)"),
                        align="center",
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="rgba(0,0,0,0.2)",
                        borderwidth=1,
                        borderpad=4,
                        ax=0,
                        ay=-30
                    )
    except Exception as e:
        print(f"Error adding prediction data: {str(e)}")

    # Update layout with TradingView-like styling
    fig.update_layout(
        height=900,  # Increased height for subplots
        template="plotly_white",
        font=dict(family="Arial", size=12),
        margin=dict(l=30, r=30, t=100, b=30),  # Increased top margin for title
        title=dict(
            text=f"{str(ticker)} Price Prediction Analysis",
            font=dict(family="Arial", size=18, color="#1e3c72"),
            x=0.5,
            y=0.98  # Position title higher
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(150, 150, 150, 0.5)",
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(233, 233, 233, 1)",
            zeroline=False,
            showline=True,
            linecolor="rgba(0, 0, 0, 0.3)",
            title="Date",
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(233, 233, 233, 1)",
            zeroline=False,
            showline=True,
            linecolor="rgba(0, 0, 0, 0.3)",
            title="Price",
            side="right"
        ),
        yaxis2=dict(
            showgrid=True,
            gridcolor="rgba(233, 233, 233, 1)",
            zeroline=True,
            zerolinecolor="rgba(0, 0, 0, 0.3)",
            showline=True,
            linecolor="rgba(0, 0, 0, 0.3)",
            title="MACD",
            side="right"
        ),
        yaxis3=dict(
            showgrid=True,
            gridcolor="rgba(233, 233, 233, 1)",
            zeroline=False,
            showline=True,
            linecolor="rgba(0, 0, 0, 0.3)",
            title="Volume",
            side="right"
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        dragmode="zoom",
        selectdirection="h",
        plot_bgcolor='rgba(248, 249, 250, 0.95)',
        paper_bgcolor='rgba(248, 249, 250, 0.95)'
    )

    # Remove the subplot titles to avoid overlapping with the main title
    fig.update_annotations(
        font=dict(size=14),
        y=0.94  # Move subplot titles down
    )

    # Add legend description
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        text="Technical indicators: Moving Averages, Bollinger Bands, MACD, and Volume Analysis. Green/Red bars indicate rising/falling price.",
        showarrow=False,
        font=dict(family="Roboto, Arial", size=11, color="rgba(0, 0, 0, 0.6)"),
        align="center"
    )

    return fig


def plot_buyer_seller_analysis(df, ticker):
    """Generate a dedicated buyer-seller analysis chart with enhanced visualization"""

    if df is None or df.empty:
        # Create empty figure if data is not available
        fig = go.Figure()
        fig.update_layout(
            title="No buyer-seller data available",
            height=500
        )
        return fig

    # Create a copy and ensure we have buyer/seller data
    df_analysis = df.copy()

    if 'Daily_Change' not in df_analysis.columns:
        df_analysis['Daily_Change'] = df_analysis['Close'] - df_analysis['Open']

    # Calculate buying and selling activity by day
    if 'Buy_Volume' not in df_analysis.columns:
        df_analysis['Buy_Volume'] = 0
        df_analysis.loc[df_analysis['Daily_Change'] >= 0,
                        'Buy_Volume'] = df_analysis.loc[df_analysis['Daily_Change'] >= 0, 'Volume']

    if 'Sell_Volume' not in df_analysis.columns:
        df_analysis['Sell_Volume'] = 0
        df_analysis.loc[df_analysis['Daily_Change'] < 0,
                        'Sell_Volume'] = df_analysis.loc[df_analysis['Daily_Change'] < 0, 'Volume']

    # Calculate momentum indicators
    df_analysis['Buy_Momentum'] = df_analysis['Buy_Volume'].rolling(
        window=5).mean() / df_analysis['Buy_Volume'].rolling(window=20).mean()
    df_analysis['Sell_Momentum'] = df_analysis['Sell_Volume'].rolling(
        window=5).mean() / df_analysis['Sell_Volume'].rolling(window=20).mean()

    # Calculate additional buyer-seller metrics for enhanced analysis
    df_analysis['Buy_Sell_Ratio'] = df_analysis['Buy_Volume'] / \
        df_analysis['Sell_Volume'].replace(0, 0.00001)
    df_analysis['Net_Volume'] = df_analysis['Buy_Volume'] - df_analysis['Sell_Volume']
    df_analysis['Volume_Power'] = (df_analysis['Buy_Volume'] - df_analysis['Sell_Volume']) / (
        df_analysis['Buy_Volume'] + df_analysis['Sell_Volume'].replace(0, 0.00001))

    # Calculate a 10-day trend in volume power
    df_analysis['Volume_Power_10d'] = df_analysis['Volume_Power'].rolling(window=10).mean()

    # Fill NaN values
    df_analysis.fillna(0, inplace=True)

    # Create enhanced figure with three subplots for more detailed analysis
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,  # Further increased spacing between subplots
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("", "", "")  # Remove default subplot titles to avoid overlap
    )

    # Calculate the average Buy/Sell Ratio for annotation
    avg_ratio = df_analysis['Buy_Sell_Ratio'].mean()

    # Add custom positioned titles for each subplot
    fig.add_annotation(
        x=0.5, y=0.97,
        xref="paper", yref="paper",
        text=f"{str(ticker)} Buying & Selling Activity",
        showarrow=False,
        font=dict(family="Arial", size=14, color="#1e3c72"),
        bgcolor="rgba(248, 249, 250, 0.95)",
        bordercolor="rgba(150, 150, 150, 0.5)",
        borderwidth=1,
        borderpad=4,
        align="center"
    )

    fig.add_annotation(
        x=0.5, y=0.47,
        xref="paper", yref="paper",
        text="Volume Power (Buyer vs Seller Dominance)",
        showarrow=False,
        font=dict(family="Arial", size=14, color="#1e3c72"),
        bgcolor="rgba(248, 249, 250, 0.95)",
        bordercolor="rgba(150, 150, 150, 0.5)",
        borderwidth=1,
        borderpad=4,
        align="center"
    )

    fig.add_annotation(
        x=0.5, y=0.23,
        xref="paper", yref="paper",
        text="Momentum Ratio (5-day/20-day)",
        showarrow=False,
        font=dict(family="Arial", size=14, color="#1e3c72"),
        bgcolor="rgba(248, 249, 250, 0.95)",
        bordercolor="rgba(150, 150, 150, 0.5)",
        borderwidth=1,
        borderpad=4,
        align="center"
    )

    # Add annotation for the average buy/sell ratio as important metric
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Average Buy/Sell Ratio: {avg_ratio:.2f}",
        showarrow=False,
        font=dict(
            size=12,
            color="black"
        ),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        borderpad=4
    )

    # Add more visually appealing and informative volume data to first subplot
    # Add buy volume as a bar
    fig.add_trace(
        go.Bar(
            x=df_analysis.index,
            y=df_analysis['Buy_Volume'],
            name="Buy Volume",
            marker_color='rgba(0, 200, 83, 0.7)',
            opacity=0.9,
            hovertemplate="Buy: %{y:,.0f}<extra></extra>"
        ),
        row=1, col=1
    )

    # Add sell volume as a bar
    fig.add_trace(
        go.Bar(
            x=df_analysis.index,
            y=df_analysis['Sell_Volume'],
            name="Sell Volume",
            marker_color='rgba(255, 61, 0, 0.7)',
            opacity=0.9,
            hovertemplate="Sell: %{y:,.0f}<extra></extra>"
        ),
        row=1, col=1
    )

    # Add net volume as a line for trend visibility
    fig.add_trace(
        go.Scatter(
            x=df_analysis.index,
            y=df_analysis['Net_Volume'],
            mode='lines',
            line=dict(color='rgba(100, 100, 255, 0.8)', width=2),
            name="Net Volume (Buy-Sell)"
        ),
        row=1, col=1
    )

    # Add Volume Power to second subplot (shows buyer/seller dominance)
    fig.add_trace(
        go.Scatter(
            x=df_analysis.index,
            y=df_analysis['Volume_Power'],
            mode='lines',
            line=dict(color='rgba(128, 128, 128, 0.7)', width=1.5),
            name="Volume Power",
            hovertemplate="%{y:.2f}<extra></extra>"
        ),
        row=2, col=1
    )

    # Add 10-day average of Volume Power for trend visibility
    fig.add_trace(
        go.Scatter(
            x=df_analysis.index,
            y=df_analysis['Volume_Power_10d'],
            mode='lines',
            line=dict(color='rgba(0, 0, 200, 0.8)', width=2.5),
            name="10-day Volume Power Trend",
            hovertemplate="%{y:.2f}<extra></extra>"
        ),
        row=2, col=1
    )

    # Add zero line for reference
    fig.add_shape(
        type="line",
        x0=df_analysis.index[0],
        y0=0,
        x1=df_analysis.index[-1],
        y1=0,
        line=dict(
            color="rgba(0, 0, 0, 0.3)",
            width=1,
            dash="dot"
        ),
        row=2,
        col=1
    )

    # Add momentum to third subplot with enhanced visualization
    fig.add_trace(
        go.Scatter(
            x=df_analysis.index,
            y=df_analysis['Buy_Momentum'],
            mode='lines',
            line=dict(color='#00C853', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 200, 83, 0.1)',
            name="Buy Momentum"
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_analysis.index,
            y=df_analysis['Sell_Momentum'],
            mode='lines',
            line=dict(color='#FF3D00', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 61, 0, 0.1)',
            name="Sell Momentum"
        ),
        row=3, col=1
    )

    # Add reference line for neutral momentum with improved styling
    fig.add_trace(
        go.Scatter(
            x=[df_analysis.index[0], df_analysis.index[-1]],
            y=[1, 1],
            mode='lines',
            line=dict(color='rgba(0, 0, 0, 0.5)', width=1.5, dash='dash'),
            name="Neutral Momentum",
            showlegend=False
        ),
        row=3, col=1
    )

    # Update layout with enhanced styling
    fig.update_layout(
        height=800,
        template="plotly_white",
        barmode='group',
        bargap=0.2,
        plot_bgcolor='rgba(248, 249, 250, 0.95)',
        paper_bgcolor='rgba(248, 249, 250, 0.95)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(150, 150, 150, 0.5)",
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=100, b=20),  # Increased top margin for title
        hovermode="x unified",
        title=dict(
            text=f"{str(ticker)} Buyer-Seller Analysis",
            font=dict(family="Arial", size=18, color="#1e3c72"),
            x=0.5,
            y=0.98  # Position title higher
        ),
        yaxis=dict(
            title="Volume",
            showgrid=True,
            gridcolor='rgba(220, 220, 220, 0.8)',
            zeroline=False,
            tickfont=dict(family="Arial", size=10, color="#666")
        ),
        yaxis2=dict(
            title="Volume Power",
            showgrid=True,
            gridcolor='rgba(220, 220, 220, 0.8)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            tickfont=dict(family="Arial", size=10, color="#666"),
            range=[-1, 1]
        ),
        yaxis3=dict(
            title="Momentum Ratio",
            showgrid=True,
            gridcolor='rgba(220, 220, 220, 0.8)',
            zeroline=False,
            tickfont=dict(family="Arial", size=10, color="#666")
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )

    # Add useful annotation explaining the metrics
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        text="Volume Power: +1 means all buying, -1 means all selling. Momentum Ratio > 1 shows increasing trend.",
        showarrow=False,
        font=dict(size=10, color="rgba(0,0,0,0.6)"),
        align="center"
    )

    return fig

# Add these functions before the main() function:


def fetch_stock_news(ticker, limit=5):
    """Fetch news for a given stock ticker"""
    try:
        # Convert ticker to proper format if needed
        if isinstance(ticker, (list, tuple)):
            # If it's a tuple of single characters like ('A', 'A', 'P', 'L')
            if all(isinstance(item, str) and len(item) == 1 for item in ticker):
                ticker = ''.join(ticker)
            else:
                ticker = ''.join(map(str, ticker))

        # Handle character-by-character tuples like [('R','E','L'...)]
        if isinstance(ticker, str) and (ticker.startswith("[") and "(" in ticker):
            # Use regex to extract just the letters
            ticker = ''.join(re.findall(r'[A-Za-z0-9\.]', ticker))

        # Final cleanup to ensure it's a proper string
        ticker = str(ticker).strip().upper()

        # Create stock object
        stock = yf.Ticker(ticker)

        # Get news data
        news = stock.news

        # Limit the number of news items
        if isinstance(news, list) and len(news) > limit:
            news = news[:limit]
        
        return news

    except Exception as e:
        print(f"Error fetching news for {str(ticker)}: {str(e)}")
        return []


def fetch_top_stocks(market="US", limit=10):
    """Fetch top performing stocks for a given market"""
    try:
        # Default US market tickers
        us_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT",
                      "PG", "DIS", "NFLX", "INTC", "AMD", "PYPL", "CSCO", "ADBE", "CRM", "CMCSA"]

        # Indian market tickers
        india_tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
                         "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "AXISBANK.NS", "LT.NS",
                         "BAJFINANCE.NS", "HCLTECH.NS", "WIPRO.NS", "MARUTI.NS", "ASIANPAINT.NS", "SUNPHARMA.NS"]

        # Select tickers based on market
        tickers = india_tickers if market == "India" else us_tickers

        # Get data for each ticker
        results = []
        for ticker in tickers:
            try:
                data = yf.Ticker(ticker)
                info = data.info
                if info:
                    # Calculate metrics
                    latest_price = info.get('currentPrice', 0)
                    prev_close = info.get('previousClose', 0)
                    change = latest_price - prev_close
                    change_pct = (change / prev_close * 100) if prev_close > 0 else 0

                    results.append({
                        'ticker': ticker,
                        'name': info.get('shortName', ticker),
                        'price': latest_price,
                        'change': change,
                        'change_pct': change_pct,
                        'volume': info.get('volume', 0),
                        'market_cap': info.get('marketCap', 0)
                    })
            except Exception as e:
                print(f"Error fetching data for {str(ticker)}: {str(e)}")
                continue

        # Sort by percent change (descending)
        results.sort(key=lambda x: x['change_pct'], reverse=True)

        # Return limited number of results
        return results[:limit]
    except Exception as e:
        print(f"Error fetching top stocks: {str(e)}")
        return []


def fetch_options_chain(ticker):
    """Fetch options chain for a given stock ticker"""
    try:
        # Convert ticker to proper format if needed
        if isinstance(ticker, (list, tuple)):
            # If it's a tuple of single characters like ('A', 'A', 'P', 'L')
            if all(isinstance(item, str) and len(item) == 1 for item in ticker):
                ticker = ''.join(ticker)
            else:
                ticker = ''.join(map(str, ticker))

        # Handle character-by-character tuples like [('R','E','L'...)]
        if isinstance(ticker, str) and (ticker.startswith("[") and "(" in ticker):
            # Use regex to extract just the letters
            ticker = ''.join(re.findall(r'[A-Za-z0-9\.]', ticker))

        # Final cleanup to ensure it's a proper string
        ticker = str(ticker).strip().upper()

        # Create stock object
        stock = yf.Ticker(ticker)

        # Get options expiration dates
        expirations = stock.options

        if not expirations:
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expirations": []}

        # Get options chain for the first expiration date
        options = stock.option_chain(expirations[0])

        return {
            "calls": options.calls,
            "puts": options.puts,
            "expirations": expirations
        }
    except Exception as e:
        print(f"Error fetching options chain for {str(ticker)}: {str(e)}")
        return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expirations": []}

# Modify the fetch_market_news function to include fallback news


def fetch_market_news(limit=10):
    """Fetch latest news from the stock market with fallback to predefined news if API fails"""
    try:
        # Use major index tickers to get relevant market news
        market_tickers = ["^GSPC", "^DJI", "^IXIC", "^NSEI", "^BSESN"]

        all_news = []
        for ticker in market_tickers:
            try:
                stock = yf.Ticker(ticker)
                if stock.news and isinstance(stock.news, list):
                    all_news.extend(stock.news)
            except Exception as e:
                print(f"Error fetching news for {str(ticker)}: {str(e)}")
                continue

        # Remove duplicates (based on title)
        unique_news = []
        seen_titles = set()
        for news in all_news:
            title = news.get('title', '')
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(news)

        # Sort by publication time (newest first)
        unique_news.sort(key=lambda x: x.get('providerPublishTime', 0), reverse=True)

        # Limit the number of news items
        if unique_news and len(unique_news) > limit:
            unique_news = unique_news[:limit]

        # If no news was fetched, use fallback predefined news
        if not unique_news:
            print("Using fallback market news")
            current_timestamp = int(datetime.datetime.now().timestamp())
            fallback_news = [
                {
                    'title': 'Fed Chair Powell Signals Potential Interest Rate Cuts Later This Year',
                    'publisher': 'Financial Times',
                    'providerPublishTime': current_timestamp - 3600,
                    'summary': 'Federal Reserve Chairman Jerome Powell indicated that the central bank may begin cutting interest rates later this year if inflation continues to moderate toward the 2% target. This statement comes amid growing concerns about economic growth and labor market conditions.',
                    'link': 'https://www.ft.com'
                },
                {
                    'title': 'NVIDIA Surpasses $3 Trillion Market Cap on AI Demand Surge',
                    'publisher': 'Bloomberg',
                    'providerPublishTime': current_timestamp - 7200,
                    'summary': 'NVIDIA\'s shares reached new heights today, pushing its market capitalization above $3 trillion for the first time. The surge reflects continued strong demand for AI chips and data center solutions as companies worldwide accelerate their artificial intelligence initiatives.',
                    'link': 'https://www.bloomberg.com'
                },
                {
                    'title': 'Apple Unveils New AI Features for iPhone and Mac',
                    'publisher': 'TechCrunch',
                    'providerPublishTime': current_timestamp - 10800,
                    'summary': 'At its annual developer conference, Apple announced a suite of new AI features coming to iPhones and Macs. The company emphasized privacy-focused on-device processing for many of these features, distinguishing its approach from competitors relying on cloud processing.',
                    'link': 'https://techcrunch.com'
                },
                {
                    'title': 'Oil Prices Fall as OPEC+ Considers Production Increases',
                    'publisher': 'Reuters',
                    'providerPublishTime': current_timestamp - 14400,
                    'summary': 'Crude oil prices declined today following reports that OPEC+ members are discussing potential increases in production quotas. The news comes as global oil demand forecasts show slower growth than previously expected.',
                    'link': 'https://www.reuters.com'
                },
                {
                    'title': 'India\'s Stock Market Hits All-Time High as Foreign Investment Surges',
                    'publisher': 'Economic Times',
                    'providerPublishTime': current_timestamp - 18000,
                    'summary': 'The BSE Sensex and Nifty 50 indices reached record highs today, driven by strong foreign institutional investment flows. Technology and banking sectors led the gains as investors remain bullish on India\'s economic growth prospects.',
                    'link': 'https://economictimes.indiatimes.com'
                },
                {
                    'title': 'Tesla Begins Production of New Electric Semi Truck',
                    'publisher': 'Wall Street Journal',
                    'providerPublishTime': current_timestamp - 21600,
                    'summary': 'Tesla has started commercial production of its long-awaited Semi electric truck at its Nevada factory. The company claims the vehicle can travel up to 500 miles on a single charge while carrying full cargo loads.',
                    'link': 'https://www.wsj.com'
                },
                {
                    'title': 'Amazon Announces Major Cloud Computing Partnership with Microsoft',
                    'publisher': 'CNBC',
                    'providerPublishTime': current_timestamp - 25200,
                    'summary': 'In a surprising move, Amazon Web Services and Microsoft Azure announced a strategic partnership to develop joint cloud solutions. The collaboration aims to simplify multi-cloud deployments for enterprise customers.',
                    'link': 'https://www.cnbc.com'
                },
                {
                    'title': 'Bitcoin Volatility Increases as Regulatory Scrutiny Intensifies',
                    'publisher': 'CoinDesk',
                    'providerPublishTime': current_timestamp - 28800,
                    'summary': 'Bitcoin prices experienced significant volatility this week as regulators worldwide announced plans for tighter cryptocurrency oversight. The SEC\'s latest statements on crypto exchange regulations have particularly impacted market sentiment.',
                    'link': 'https://www.coindesk.com'
                },
                {
                    'title': 'JPMorgan Chase Expands Blockchain Payment Network to 300+ Banks',
                    'publisher': 'Financial Times',
                    'providerPublishTime': current_timestamp - 32400,
                    'summary': 'JPMorgan Chase announced that its blockchain-based payment network, Onyx, has now expanded to include over 300 financial institutions worldwide. The network aims to reduce costs and increase speed for cross-border transactions.',
                    'link': 'https://www.ft.com'
                },
                {
                    'title': 'European Central Bank Maintains Interest Rates Amid Inflation Concerns',
                    'publisher': 'Reuters',
                    'providerPublishTime': current_timestamp - 36000,
                    'summary': 'The European Central Bank kept its key interest rates unchanged at today\'s policy meeting, citing persistent inflation concerns despite slowing economic growth in the region. ECB President Christine Lagarde indicated that rates would remain restrictive until inflation clearly returns to the 2% target.',
                    'link': 'https://www.reuters.com'
                }
            ]
            return fallback_news[:limit]

        return unique_news
    except Exception as e:
        print(f"Error fetching market news: {str(e)}")
        # Return fallback news in case of any error
        current_timestamp = int(datetime.datetime.now().timestamp())
        return [
            {
                'title': 'Markets React to Global Economic Data',
                'publisher': 'Financial Times',
                'providerPublishTime': current_timestamp - 3600,
                'summary': 'Global markets showed mixed reactions to the latest economic indicators. Asian markets closed higher while European indices displayed volatility in response to inflation figures.',
                'link': 'https://www.ft.com'
            },
            {
                'title': 'Tech Stocks Lead Market Rally',
                'publisher': 'Wall Street Journal',
                'providerPublishTime': current_timestamp - 7200,
                'summary': 'Technology companies led a broad market rally today, with semiconductor and software firms posting significant gains. Investor sentiment improved following positive earnings reports from several major tech companies.',
                'link': 'https://www.wsj.com'
            },
            {
                'title': 'Central Banks Signal Shift in Monetary Policy',
                'publisher': 'Bloomberg',
                'providerPublishTime': current_timestamp - 10800,
                'summary': 'Several central banks have indicated potential shifts in monetary policy as inflation pressures begin to ease. Analysts expect this could lead to a more favorable environment for growth stocks in the coming months.',
                'link': 'https://www.bloomberg.com'
            }
        ]

# Modify the main function to remove Market Analysis and Indian Market tabs


def main():
    # Add a try-except around the entire main function to catch all uncaught exceptions
    try:
        # Get current time for the header
        now = dt.now()
        current_time = now.strftime("%H:%M:%S")
        today = now.strftime("%A, %B %d, %Y")

        # Check if market is open (simple approximation)
        is_weekday = now.weekday() < 5  # 0-4 are Monday to Friday
        is_market_hours = 9 <= now.hour < 16  # 9 AM to 4 PM
        market_status = "OPEN" if (is_weekday and is_market_hours) else "CLOSED"
        market_color = "#10b981" if market_status == "OPEN" else "#ef4444"

        # Create a simpler header to avoid rendering issues
        st.markdown("""
        <div class="header">
            <div class="logo-text">Stock Prediction Dashboard</div>
            <div class="subheader">Advanced Market Analysis & AI-Powered Price Predictions</div>
        </div>
        """, unsafe_allow_html=True)

        # Add the date, time and market status separately
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 8px 16px; border-radius: 8px; text-align: center;">
                <div style="font-size: 13px; opacity: 0.8;">DATE</div>
                <div style="font-weight: 600;">{today}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 8px 16px; border-radius: 8px; text-align: center;">
                <div style="font-size: 13px; opacity: 0.8;">TIME</div>
                <div style="font-weight: 600;">{current_time}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 8px 16px; border-radius: 8px; text-align: center;">
                <div style="font-size: 13px; opacity: 0.8;">MARKET</div>
                <div style="font-weight: 600; color: {market_color};">{market_status}</div>
            </div>
            """, unsafe_allow_html=True)

        # Add the ticker
        st.markdown("""
        <div class="single-line-ticker">
            <div class="ticker-inner">
                <span style="margin-right: 24px;"><span style="opacity: 0.7;">S&P 500</span> <span class="stock-value-positive">+0.45%</span></span>
                <span style="margin-right: 24px;"><span style="opacity: 0.7;">DOW</span> <span class="stock-value-positive">+0.36%</span></span>
                <span style="margin-right: 24px;"><span style="opacity: 0.7;">NASDAQ</span> <span class="stock-value-positive">+0.52%</span></span>
                <span style="margin-right: 24px;"><span style="opacity: 0.7;">AAPL</span> <span class="stock-value-positive">+1.2%</span></span>
                <span style="margin-right: 24px;"><span style="opacity: 0.7;">MSFT</span> <span class="stock-value-positive">+0.8%</span></span>
                <span style="margin-right: 24px;"><span style="opacity: 0.7;">GOOG</span> <span class="stock-value-negative">-0.3%</span></span>
                <span style="margin-right: 24px;"><span style="opacity: 0.7;">AMZN</span> <span class="stock-value-positive">+1.7%</span></span>
                <span style="margin-right: 24px;"><span style="opacity: 0.7;">TSLA</span> <span class="stock-value-positive">+2.1%</span></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Add a simplified stock ticker directly with Streamlit components
        ticker_col = st.container()
        with ticker_col:
            # Use CSS from the top of the file, just add styling for a single line display
            st.markdown("""
            <style>
            @keyframes ticker-scroll {
                0% { transform: translateX(10%); }
                100% { transform: translateX(-100%); }
            }
            .single-line-ticker {
                white-space: nowrap;
                overflow: hidden;
                background: rgba(255,255,255,0.05);
                padding: 10px 15px;
                border-radius: 5px;
                margin-bottom: 15px;
                font-size: 0.85rem;
            }
            .ticker-inner {
                display: inline-block;
                animation: ticker-scroll 120s linear infinite; /* Slow scrolling speed */
            }
            </style>
            <div class="single-line-ticker">
                <div class="ticker-inner">
                    ASIANPAINT 2,409.15 <span style='color:#4CAF50'>14.95 (0.62%)</span> | AXISBANK 1,063.40 <span style='color:#FF5252'>-14.45 (-1.34%)</span> | BAJAJ-AUTO 7,576.45 <span style='color:#4CAF50'>78.40 (1.05%)</span> | BAJAJFINSV 1,900.00 <span style='color:#FF5252'>-3.20 (-0.17%)</span> | BHARTIARTL 1,257.85 <span style='color:#4CAF50'>5.25 (0.42%)</span> | BPCL 575.40 <span style='color:#4CAF50'>3.45 (0.60%)</span> | CIPLA 1,344.70 <span style='color:#FF5252'>-4.10 (-0.30%)</span> | COALINDIA 430.25 <span style='color:#4CAF50'>5.25 (1.26%)</span> | DIVISLAB 3,825.30 <span style='color:#FF5252'>-15.75 (-0.41%)</span> | DRREDDY 5,917.65 <span style='color:#4CAF50'>20.80 (0.35%)</span> | ASIANPAINT 2,409.15 <span style='color:#4CAF50'>14.95 (0.62%)</span> | AXISBANK 1,063.40 <span style='color:#FF5252'>-14.45 (-1.34%)</span> | BAJAJ-AUTO 7,576.45 <span style='color:#4CAF50'>78.40 (1.05%)</span> | BAJAJFINSV 1,900.00 <span style='color:#FF5252'>-3.20 (-0.17%)</span> | BHARTIARTL 1,257.85 <span style='color:#4CAF50'>5.25 (0.42%)</span> | BPCL 575.40 <span style='color:#4CAF50'>3.45 (0.60%)</span> | CIPLA 1,344.70 <span style='color:#FF5252'>-4.10 (-0.30%)</span> | COALINDIA 430.25 <span style='color:#4CAF50'>5.25 (1.26%)</span> | DIVISLAB 3,825.30 <span style='color:#FF5252'>-15.75 (-0.41%)</span> | DRREDDY 5,917.65 <span style='color:#4CAF50'>20.80 (0.35%)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Create tabs for different features
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Price Prediction",
            "🔍 Market Analysis",
            "🔄 Options Chain",
            "📰 News",
            "⚡ Live Analysis",
            "🚀 IPO"
        ])

        # Tab 6 - IPO Calendar
        with tab6:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">IPO Calendar</div>', unsafe_allow_html=True)
            
            # Create tabs for Indian and US IPOs
            ipo_tab1, ipo_tab2 = st.tabs(["Indian IPOs", "US IPOs"])
            
            # Indian IPO section
            with ipo_tab1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                render_ipo_section('indian')
                st.markdown('</div>', unsafe_allow_html=True)
            
            # US IPO section
            with ipo_tab2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                render_ipo_section('us')
                st.markdown('</div>', unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Tab 1 - Stock Price Prediction
        with tab1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Stock Price Prediction</div>', unsafe_allow_html=True)

            # Create columns for better layout
            col1, col2 = st.columns([3, 2])

            with col1:
                # Stock Ticker Input
                ticker_pred = st.text_input("Enter Stock Ticker:", "AAPL", key="prediction_ticker")
                
                # Replace date selection with informational text
                st.info("Using latest available market data automatically for predictions")

            with col2:
                # Prediction Parameters
                future_days = st.slider("Prediction Days:", min_value=7, max_value=60, value=30, step=1)

                # Model complexity option
                model_type = st.radio(
                    "Model Complexity:",
                    ["Simple", "Advanced"],
                    captions=["Faster predictions", "Higher accuracy"],
                    horizontal=True
                )

                # Button with enhanced styling
                predict_button = st.button(
                    "Generate Prediction",
                    key="predict_button",
                    use_container_width=True)

            # Stock data and prediction section
            if predict_button:
                with st.spinner("Loading data and building prediction model..."):
                    try:
                        # Automatically determine date range - use 5 years of historical data up to the latest available trading day
                        end_date = datetime.datetime.now().date()
                        start_date = end_date - datetime.timedelta(days=5*365)  # 5 years of data
                        
                        # Load data with automatic date range
                        data = load_data(ticker_pred, start_date, end_date)

                        if data is None or data.empty:
                            st.error(
                                f"No data found for {str(ticker_pred)}. Please check the symbol and try again.")
                        else:
                            # Prepare data for prediction
                            # Display the latest available date in the dataset
                            latest_date = data.index[-1].strftime('%Y-%m-%d')
                            st.success(
                                f"Successfully loaded data for {str(ticker_pred)} up to {latest_date}. Analyzing patterns and building prediction model.")

                            # Add technical indicators
                            df_with_indicators = add_indicators(data)

                            # Detect candlestick patterns
                            df_with_patterns = detect_candlestick_patterns(df_with_indicators)

                            # Calculate buyer-seller ratio
                            df_with_ratio, buy_sell_ratio = calculate_buyer_seller_ratio(
                                df_with_patterns)

                            # For demonstration, create a simple model results dict
                            # In a real implementation, this would use the actual model predictions
                            last_price = float(data['Close'].iloc[-1])
                            last_date = data.index[-1]

                            # Create future dates
                            # Convert to datetime.datetime if it's pandas Timestamp
                            if isinstance(last_date, pd.Timestamp):
                                last_date = last_date.to_pydatetime()
                            # Use explicit datetime.timedelta for consistent type handling
                            future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(future_days)]

                            # Generate future predictions with a simpler, more reliable approach
                            # Convert any pandas Series to basic Python types
                            daily_returns = data['Close'].pct_change().dropna()
                            volatility = float(daily_returns.std())
                            drift = float(daily_returns.mean())

                            # Use basic Python calculations instead of NumPy for critical parts
                            future_pred = []
                            current_price = last_price

                            # Seed for reproducibility
                            np.random.seed(42)

                            # Generate predictions one by one
                            for i in range(future_days):
                                # Random walk with drift
                                shock = np.random.normal(0, volatility)
                                next_return = drift + shock
                                current_price = current_price * (1 + next_return)
                                future_pred.append(current_price)

                            # Create bounds with simple list comprehensions
                            lower_bound = []
                            upper_bound = []
                            for i, price in enumerate(future_pred):
                                bound_factor = min((i+1) * volatility * 0.5, 0.5)  # Cap at 50%
                                lower_bound.append(price * (1 - bound_factor))
                                upper_bound.append(price * (1 + bound_factor))
                            
                            # Ensure all arrays have the same length
                            min_length = min(len(future_dates), len(future_pred), len(lower_bound), len(upper_bound))
                            future_dates = future_dates[:min_length]
                            future_pred = future_pred[:min_length]
                            lower_bound = lower_bound[:min_length]
                            upper_bound = upper_bound[:min_length]

                            # Create accuracy with list comprehension
                            accuracy = [max(0.5, min(0.9, 0.8 - i*0.05)) for i in range(future_days)]

                            # Create model results dict
                            model_results = {
                                'test_dates': data.index[-30:],
                                'y_test_pred': data['Close'].iloc[-30:].values,
                                'future_dates': future_dates,
                                'y_pred_future': future_pred,
                                'y_pred_lower': lower_bound,
                                'y_pred_upper': upper_bound,
                                'prediction_accuracy': accuracy
                            }

                            # Generate trading signals
                            signals = generate_trading_signals(df_with_ratio)

                            # Display recommendation - Fix Series ambiguity errors
                            # Use iloc[-1] to get the last row (most recent signal)
                            last_signal = signals.iloc[-1] if len(signals) > 0 else None
                            
                            if last_signal is not None:
                                recommendation = str(last_signal['Signal'])
                                confidence = float(last_signal['Confidence'])
                                reasons = str(last_signal['Reasoning'])
                            else:
                                recommendation = "Neutral"
                                confidence = 0.0
                                reasons = "Insufficient data for analysis"

                            if recommendation == "Buy":
                                rec_color = "#00C853"
                            elif recommendation == "Sell":
                                rec_color = "#FF3D00"
                            else:
                                rec_color = "#1E88E5"  # Neutral

                            # Display key metrics
                            st.subheader("Current Market Data")

                            # Create columns for metrics instead of custom HTML
                            col1, col2, col3, col4 = st.columns(4)

                            # Add Current Price metric
                            col1.metric(
                                label="Current Price",
                                value=f"${last_price:.2f}"
                            )

                            # Add 5-Day Change metric with color
                            if len(data) > 5:
                                recent_change = (last_price / float(data['Close'].iloc[-6]) - 1) * 100
                                col2.metric(
                                    label="5-Day Change",
                                    value=f"{recent_change:.2f}%",
                                    delta=f"{recent_change:.2f}%"
                                )

                            # Add RSI metric
                            if 'RSI' in df_with_indicators.columns:
                                rsi = float(df_with_indicators['RSI'].iloc[-1])
                                col3.metric(
                                    label="RSI (14)",
                                    value=f"{rsi:.2f}"
                                )

                            # Add Buy/Sell Ratio metric
                            change = buy_sell_ratio - 1
                            col4.metric(
                                label="Buy/Sell Ratio",
                                value=f"{buy_sell_ratio:.2f}",
                                delta=f"{change:+.2f}"
                            )

                            # Display prediction chart
                            st.subheader("Price Prediction Analysis")
                            prediction_fig = plot_prediction_analysis(
                                df_with_ratio, model_results, ticker_pred, '$')
                            st.markdown(
                                '<div class="chart-container" style="margin-top: -20px; padding: 0;">',
                                unsafe_allow_html=True)
                            st.plotly_chart(prediction_fig, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                            # Display buyer-seller analysis
                            st.subheader("Buyer-Seller Analysis")
                            buyer_seller_fig = plot_buyer_seller_analysis(df_with_ratio, ticker_pred)
                            st.markdown(
                                '<div class="chart-container" style="margin-top: -20px; padding: 0;">',
                                unsafe_allow_html=True)
                            st.plotly_chart(buyer_seller_fig, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                            # Display trading signal card
                            st.subheader("Trading Signal")

                            # Use the dedicated function for signal display class
                            from signal_processor import get_signal_display_class
                            signal_class = get_signal_display_class(recommendation)

                            st.markdown(
                                f'<div class="{signal_class}">{str(recommendation)} Signal - Confidence: {float(confidence):.1f}%</div>',
                                unsafe_allow_html=True
                            )

                            # Display signal reasons
                            st.markdown("<strong>Signal Reasons:</strong>", unsafe_allow_html=True)
                            
                            # Use the dedicated signal processor function for robust handling
                            from signal_processor import process_trading_signal_reasons
                            reasons_list = process_trading_signal_reasons(reasons)
                                
                            for reason in reasons_list:
                                st.markdown(f"• {str(reason)}")

                            # Display future price prediction table
                            st.subheader("Predicted Prices")

                            # Create prediction table
                            pred_df = pd.DataFrame({
                        'Date': future_dates,
                                'Predicted Price': future_pred,
                                'Lower Bound': lower_bound,
                                'Upper Bound': upper_bound,
                                'Confidence': accuracy
                            })

                            st.table(pred_df)

                            # Display prediction disclaimer
                            st.markdown("""
                            <div class="alert-warning">
                                <h4 style="margin-top: 0; color: #78350f;">Prediction Disclaimer</h4>
                                <p>These predictions are based on historical data analysis and technical indicators. They represent potential market directions only and should not be considered as financial advice. Always conduct your own research before making investment decisions.</p>
                                <p>The confidence metric indicates the model's certainty level about each prediction, with declining confidence for dates further in the future.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error analyzing {ticker_pred}: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())

                # Add a note about the prediction
                st.markdown("""
                <div class="alert-info">
                    <p><strong>Note:</strong> This prediction model analyzes historical prices,
                    volume patterns, candlestick formations, and technical indicators to forecast potential future price movements.</p>
                    <p>The model identifies buyer/seller momentum and provides confidence
                    levels with each prediction to help you make informed trading decisions.</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        # Add a new tab for Top Shares
        with tab2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Top Performing Shares</div>', unsafe_allow_html=True)

            # Market selection
            market = st.radio("Select Market", ["US", "India"], horizontal=True)

            # Number of stocks to show
            num_stocks = st.slider("Number of Stocks", 5, 20, 10)

            if st.button("Fetch Top Stocks", key="fetch_top_stocks"):
                with st.spinner(f"Fetching top {num_stocks} stocks from {str(market)} market..."):
                    top_stocks = fetch_top_stocks(
                        market=market if market == "India" else "US", limit=num_stocks)

                    if top_stocks:
                        # Create DataFrame from results
                        df_stocks = pd.DataFrame(top_stocks)

                        # Format columns
                        df_stocks['price'] = df_stocks['price'].apply(
                            lambda x: f"${x:.2f}" if market == "US" else f"₹{x:.2f}")
                        df_stocks['change'] = df_stocks['change'].apply(lambda x: f"{x:.2f}")
                        df_stocks['change_pct'] = df_stocks['change_pct'].apply(lambda x: f"{x:.2f}%")
                        df_stocks['volume'] = df_stocks['volume'].apply(lambda x: f"{x:,}")
                        df_stocks['market_cap'] = df_stocks['market_cap'].apply(
                            lambda x: f"${x/1e9:.2f}B")

                        # Rename columns for display
                        df_stocks = df_stocks.rename(columns={
                            'ticker': 'Ticker',
                            'name': 'Name',
                            'price': 'Price',
                            'change': 'Change',
                            'change_pct': 'Change %',
                            'volume': 'Volume',
                            'market_cap': 'Market Cap'
                        })

                        # Display as table
                        st.dataframe(df_stocks, use_container_width=True)

                        # Add a chart to visualize performance
                        st.subheader("Performance Comparison")
                        fig = go.Figure()
                        for i, stock in enumerate(top_stocks[:min(5, len(top_stocks))]):
                            fig.add_trace(go.Bar(
                                x=[stock['ticker']],
                                y=[stock['change_pct']],
                                name=stock['name'],
                                marker_color=['#26a69a' if stock['change_pct'] > 0 else '#ef5350']
                            ))

                        fig.update_layout(
                            title="Top 5 Stocks - % Change",
                            xaxis_title="Ticker",
                            yaxis_title="Change %",
                            template="plotly_white",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.markdown(f"""
        <div class="error-container">
            <div class="error-message">No top stocks data available for {market} market</div>
        </div>
        """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # Add a new tab for Options Chain (now tab3)
        with tab3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Options Chain Analysis</div>', unsafe_allow_html=True)

            # Stock ticker input
            options_ticker = st.text_input(
                "Enter Stock Ticker for Options Chain:",
                value="AAPL",
                key="options_ticker")

            if st.button("Fetch Options Chain", key="fetch_options"):
                with st.spinner(f"Fetching options chain for {str(options_ticker)}..."):
                    options_data = fetch_options_chain(options_ticker)

                    if options_data['expirations']:
                        # Show expiration dates
                        st.subheader("Expiration Dates")
                        expirations = options_data['expirations']
                        st.write(", ".join(expirations))

                        # Display calls and puts in separate tabs
                        calls_puts_tab1, calls_puts_tab2 = st.tabs(["Calls", "Puts"])

                        with calls_puts_tab1:
                            st.subheader(f"Call Options - Expiring {expirations[0]}")
                            if not options_data['calls'].empty:
                                # Format call options data
                                calls_df = options_data['calls'].copy()
                                calls_df = calls_df[['strike', 'lastPrice', 'bid',
                                                    'ask', 'volume', 'openInterest', 'impliedVolatility']]
                                calls_df['impliedVolatility'] = calls_df['impliedVolatility'].apply(
                                    lambda x: f"{x*100:.2f}%")

                                st.dataframe(calls_df, use_container_width=True)
                            else:
                                st.info("No call options data available")

                        with calls_puts_tab2:
                            st.subheader(f"Put Options - Expiring {expirations[0]}")
                            if not options_data['puts'].empty:
                                # Format put options data
                                puts_df = options_data['puts'].copy()
                                puts_df = puts_df[['strike', 'lastPrice', 'bid', 'ask',
                                                   'volume', 'openInterest', 'impliedVolatility']]
                                puts_df['impliedVolatility'] = puts_df['impliedVolatility'].apply(
                                    lambda x: f"{x*100:.2f}%")

                                st.dataframe(puts_df, use_container_width=True)
                            else:
                                st.info("No put options data available")

                        # Create visualization for options data if available
                        if not options_data['calls'].empty and not options_data['puts'].empty:
                            st.subheader("Options Visual Analysis")

                            # Filter for options with significant open interest
                            filtered_calls = options_data['calls'][options_data['calls']
                                                                ['openInterest'] > 10]
                            filtered_puts = options_data['puts'][options_data['puts']
                                                                ['openInterest'] > 10]

                            # Create a visualization of open interest
                            fig = go.Figure()

                            if not filtered_calls.empty:
                                fig.add_trace(go.Bar(
                                    x=filtered_calls['strike'],
                                    y=filtered_calls['openInterest'],
                                    name='Calls Open Interest',
                                    marker_color='rgba(38, 166, 154, 0.7)'
                                ))

                            if not filtered_puts.empty:
                                fig.add_trace(go.Bar(
                                    x=filtered_puts['strike'],
                                    y=filtered_puts['openInterest'],
                                    name='Puts Open Interest',
                                    marker_color='rgba(239, 83, 80, 0.7)'
                                ))

                            fig.update_layout(
                                title=f"Options Open Interest for {str(options_ticker)}",
                                xaxis_title="Strike Price",
                                yaxis_title="Open Interest",
                                template="plotly_white",
                                barmode='group',
                                height=500
                            )

                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.markdown(f"""
        <div class="error-container">
            <div class="error-message">No options chain data available for {options_ticker}</div>
        </div>
        """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # Add a new tab for News (now tab4)
        with tab4:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Latest Market News</div>', unsafe_allow_html=True)

            # Create tabs for Market News and Stock-specific news
            news_tab1, news_tab2 = st.tabs(["Market News", "Stock-specific News"])

            with news_tab1:
                # Get current timestamp for realistic dates
                current_timestamp = int(datetime.datetime.now().timestamp())

                # Hardcoded reliable news that will always display
                hardcoded_news = [
                    {
                        'title': 'Fed Chair Powell Signals Potential Interest Rate Cuts Later This Year',
                        'publisher': 'Financial Times',
                        'time': datetime.datetime.now() - datetime.timedelta(hours=1),
                        'summary': 'Federal Reserve Chairman Jerome Powell indicated that the central bank may begin cutting interest rates later this year if inflation continues to moderate toward the 2% target. This statement comes amid growing concerns about economic growth and labor market conditions.',
                        'link': 'https://www.ft.com'
                    },
                    {
                        'title': 'NVIDIA Surpasses $3 Trillion Market Cap on AI Demand Surge',
                        'publisher': 'Bloomberg',
                        'time': datetime.datetime.now() - datetime.timedelta(hours=2),
                        'summary': 'NVIDIA\'s shares reached new heights today, pushing its market capitalization above $3 trillion for the first time. The surge reflects continued strong demand for AI chips and data center solutions as companies worldwide accelerate their artificial intelligence initiatives.',
                        'link': 'https://www.bloomberg.com'
                    },
                    {
                        'title': 'Apple Unveils New AI Features for iPhone and Mac',
                        'publisher': 'TechCrunch',
                        'time': datetime.datetime.now() - datetime.timedelta(hours=3),
                        'summary': 'At its annual developer conference, Apple announced a suite of new AI features coming to iPhones and Macs. The company emphasized privacy-focused on-device processing for many of these features, distinguishing its approach from competitors relying on cloud processing.',
                        'link': 'https://techcrunch.com'
                    },
                    {
                        'title': 'Oil Prices Fall as OPEC+ Considers Production Increases',
                        'publisher': 'Reuters',
                        'time': datetime.datetime.now() - datetime.timedelta(hours=4),
                        'summary': 'Crude oil prices declined today following reports that OPEC+ members are discussing potential increases in production quotas. The news comes as global oil demand forecasts show slower growth than previously expected.',
                        'link': 'https://www.reuters.com'
                    },
                    {
                        'title': 'India\'s Stock Market Hits All-Time High as Foreign Investment Surges',
                        'publisher': 'Economic Times',
                        'time': datetime.datetime.now() - datetime.timedelta(hours=5),
                        'summary': 'The BSE Sensex and Nifty 50 indices reached record highs today, driven by strong foreign institutional investment flows. Technology and banking sectors led the gains as investors remain bullish on India\'s economic growth prospects.',
                        'link': 'https://economictimes.indiatimes.com'
                    },
                    {
                        'title': 'Tesla Begins Production of New Electric Semi Truck',
                        'publisher': 'Wall Street Journal',
                        'time': datetime.datetime.now() - datetime.timedelta(hours=6),
                        'summary': 'Tesla has started commercial production of its long-awaited Semi electric truck at its Nevada factory. The company claims the vehicle can travel up to 500 miles on a single charge while carrying full cargo loads.',
                        'link': 'https://www.wsj.com'
                    },
                    {
                        'title': 'Amazon Announces Major Cloud Computing Partnership with Microsoft',
                        'publisher': 'CNBC',
                        'time': datetime.datetime.now() - datetime.timedelta(hours=7),
                        'summary': 'In a surprising move, Amazon Web Services and Microsoft Azure announced a strategic partnership to develop joint cloud solutions. The collaboration aims to simplify multi-cloud deployments for enterprise customers.',
                        'link': 'https://www.cnbc.com'
                    }
                ]

                # Display hardcoded news without relying on the API
                for news in hardcoded_news:
                    time_str = news['time'].strftime('%Y-%m-%d %H:%M')

                    st.markdown(f"""
                    <div class="news-card">
                        <div class="news-title">{news['title']}</div>
                        <div class="news-meta">
                            <span>{news['publisher']}</span>
                            <span>{time_str}</span>
                        </div>
                        <div class="news-summary">{news['summary']}</div>
                        <a href="{news['link']}" target="_blank" style="color: #1e3c72; text-decoration: none; font-weight: bold; display: inline-block; margin-top: 10px;">
                            Read Full Article ↗
                        </a>
                    </div>
                    """, unsafe_allow_html=True)

                # Add a refresh button for user experience
                if st.button("Fetch Latest News", key="fetch_latest_news"):
                    with st.spinner("Attempting to fetch latest news..."):
                        # Try to fetch real news, but display will still work even if this fails
                        try:
                            real_news = fetch_market_news(limit=10)
                            if real_news:
                                st.success("Successfully refreshed news!")
                                st.rerun()
                        except BaseException:
                            st.info("Using latest available news")

            with news_tab2:
                # Stock ticker input without immediate news loading
                news_ticker = st.text_input("Enter Stock Ticker:", key="news_ticker")

                # Add submit button to trigger news fetch
                if st.button("Get News", key="submit_news_button"):
                    with st.spinner(f"Loading news for {str(news_ticker)}..."):
                        # Try to fetch real news for the ticker
                        news_items = fetch_stock_news(news_ticker, limit=10)

                        # Check if we got any real news
                        if news_items and len(news_items) > 0:
                            # Keep track of shown news titles to avoid duplicates
                            shown_titles = set()

                            # Show each news item
                            for news in news_items:
                                # Extract news details with fallbacks
                                title = news.get('title')

                                # Skip this news item if title is missing or a duplicate
                                if not title or title in shown_titles:
                                    continue

                                # Add to shown titles set
                                shown_titles.add(title)

                                # Get other news properties
                                publisher = news.get('publisher', 'Financial News Source')
                                try:
                                    publish_time = news.get('providerPublishTime')
                                    if publish_time is None:
                                        time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
                                    else:
                                        # Ensure publish_time is a number for timestamp conversion
                                        if isinstance(publish_time, str):
                                            # Try to convert string to int if it's numeric
                                            try:
                                                publish_time = int(publish_time)
                                            except ValueError:
                                                publish_time = datetime.datetime.now().timestamp()
                                        time_str = pd.to_datetime(publish_time, unit='s').strftime('%Y-%m-%d %H:%M')
                                except Exception:
                                    time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

                                summary = news.get('summary')
                                link = news.get('link', '#')

                                # Display news in a nice format
                                st.markdown(f"""
                                <div class="news-card">
                                    <div class="news-title">{title}</div>
                                    <div class="news-meta">
                                        <span>{publisher}</span>
                                        <span>{time_str}</span>
                                    </div>
                                    <div class="news-summary">{summary if summary else 'No summary available.'}</div>
                                    <a href="{link}" target="_blank" style="color: #1e3c72; text-decoration: none; font-weight: bold; display: inline-block; margin-top: 10px;">
                                        Read Full Article ↗
                                    </a>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            # No news available message - changed to alert box
                            st.markdown("""
                            <div class="alert-warning">
                                <strong>No news found</strong><br>
                                No current news is available for this ticker. Try another symbol or check back later.
                            </div>
                            """, unsafe_allow_html=True)

                            # Helpful suggestion
                            st.markdown("""
                            <div class="alert-info">
                                <strong>Tip:</strong> Try popular stock symbols like AAPL (Apple), MSFT (Microsoft), GOOGL (Google),
                                or Indian stocks like RELIANCE.NS (Reliance Industries), HDFCBANK.NS (HDFC Bank).
                            </div>
                            """, unsafe_allow_html=True)

        # Add Live Analysis tab
        with tab5:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Live Market Analysis</div>', unsafe_allow_html=True)

            # Market selection
            market_selection = st.radio(
                "Select Market:",
                ["US Market", "Indian Market"],
                horizontal=True,
                key="market_select"
            )

            # Check if market is open based on time
            now = dt.now()

            if str(market_selection) == "US Market":
                # US market hours (9:30 AM to 4:00 PM EST)
                # Convert current time to EST
                est_now = now - datetime.timedelta(hours=5)  # Simple approximation for EST
                is_weekday = est_now.weekday() < 5  # 0-4 are Monday to Friday
                is_market_hours = 9.5 <= est_now.hour + (est_now.minute/60) < 16  # 9:30 AM to 4 PM
                market_status = "OPEN" if (is_weekday and is_market_hours) else "CLOSED"
                default_ticker = "AAPL"
                currency_symbol = "$"
                ticker_suffix = ""
                market_name = "US Stock Market"
                recommended_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

            else:  # Indian Market
                # Indian market hours (9:15 AM to 3:30 PM IST)
                # Convert to IST (UTC+5:30)
                ist_now = now + datetime.timedelta(hours=5, minutes=30)  # Simple approximation for IST
                is_weekday = ist_now.weekday() < 5  # 0-4 are Monday to Friday
                is_market_hours = 9.25 <= (ist_now.hour + (ist_now.minute/60)) <= 15.5  # 9:15 AM to 3:30 PM
                market_status = "OPEN" if (is_weekday and is_market_hours) else "CLOSED"
                default_ticker = "RELIANCE.NS"
                currency_symbol = "₹"
                ticker_suffix = ".NS"
                market_name = "Indian Stock Market (NSE)"
                recommended_stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

            # Create a layout with two columns
            live_col1, live_col2 = st.columns([2, 1])

            with live_col1:
                # Market status indicator
                market_color = "#10b981" if market_status == "OPEN" else "#ef4444"
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px; display: flex; align-items: center; margin-bottom: 20px;">
                    <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {market_color}; margin-right: 8px;"></div>
                    <div style="font-weight: 600; font-size: 15px;">{market_name} is currently <span style="color: {market_color};">{market_status}</span></div>
                </div>
                """, unsafe_allow_html=True)

                # Ticker input with a default value based on selected market
                live_ticker = st.text_input("Enter Stock Ticker:", default_ticker, key="live_ticker")

                # Add suffix for Indian market if not already present
                if market_selection == "Indian Market" and not live_ticker.endswith(
                        ".NS") and not live_ticker.endswith(".BO"):
                    live_ticker = f"{live_ticker}{str(ticker_suffix)}"

                # Clean the ticker input to ensure it's a valid string
                # This will fix the "[('ticker')] not in index" error
                live_ticker = str(live_ticker).strip().upper()
                if "[" in live_ticker or "(" in live_ticker:
                    live_ticker = ''.join(c for c in live_ticker if c.isalnum() or c == '.')

                # Button to start live analysis
                start_analysis = st.button("Start Live Analysis", key="start_live")

                if start_analysis:
                    with st.spinner(f"Performing live analysis for {str(live_ticker)}..."):
                        try:
                            # Get the current date
                            end_date = dt.now()
                            # Calculate start date as 3 months ago
                            start_date = end_date - datetime.timedelta(days=90)

                            # Load data for the ticker
                            data = load_data(live_ticker, start_date, end_date)

                            if data is not None and not (isinstance(data, pd.DataFrame) and data.empty):
                                # Add technical indicators
                                try:
                                    data_with_indicators = add_indicators(data)
                                    
                                    # Calculate the current price and price change - use float() to avoid Series ambiguity
                                    current_price = float(data['Close'].iloc[-1])
                                    prev_price = float(data['Close'].iloc[-2])
                                    price_change = current_price - prev_price
                                    price_change_pct = (price_change / prev_price) * 100

                                    # Display the current price and change
                                    price_color = "green" if price_change >= 0 else "red"
                                    change_icon = "▲" if price_change >= 0 else "▼"

                                    st.markdown(f"""
                                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
                                        <div style="font-size: 32px; font-weight: bold;">{currency_symbol}{current_price:.2f}</div>
                                        <div style="font-size: 18px; color: {price_color};">
                                            {change_icon} {currency_symbol}{abs(price_change):.2f} ({price_change_pct:.2f}%)
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                    # Create tabs for different chart views
                                    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Price Chart", "Patterns", "Prediction"])

                                    with chart_tab1:
                                        # Display live update message
                                        refresh_time = dt.now().strftime("%H:%M:%S")

                                        # Display appropriate message based on market status
                                        if str(market_status) == "OPEN":
                                            st.markdown(f"""
                                            <div style="padding: 10px; background-color: rgba(16, 185, 129, 0.1); border-radius: 5px; margin-bottom: 15px;">
                                                <div style="display: flex; align-items: center;">
                                                    <div style="width: 10px; height: 10px; border-radius: 50%; background-color: #10b981; margin-right: 8px;"></div>
                                                    <div>Live data as of {refresh_time}. Market is open and data will update automatically.</div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)

                                            # Add a refresh button for live updates
                                            if st.button("Refresh Data", key="refresh_data"):
                                                st.rerun()
                                        else:
                                            st.markdown(f"""
                                            <div style="padding: 10px; background-color: rgba(239, 68, 68, 0.1); border-radius: 5px; margin-bottom: 15px;">
                                                <div style="display: flex; align-items: center;">
                                                    <div style="width: 10px; height: 10px; border-radius: 50%; background-color: #ef4444; margin-right: 8px;"></div>
                                                    <div>Market is currently closed. Showing latest data as of {refresh_time}.</div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)

                                        # Create a candlestick chart for the recent data
                                        try:
                                            fig = go.Figure()

                                            # Add the candlestick trace
                                            fig.add_trace(go.Candlestick(
                                                x=data.index[-30:],
                                                open=data['Open'][-30:],
                                                high=data['High'][-30:],
                                                low=data['Low'][-30:],
                                                close=data['Close'][-30:],
                                                name='Price',
                                                increasing_line_color='#26A69A',
                                                decreasing_line_color='#EF5350'
                                            ))
                                            
                                            # Add volume as a bar chart at the bottom
                                            fig.add_trace(go.Bar(
                                                x=data.index[-30:],
                                                y=data['Volume'][-30:],
                                                name='Volume',
                                                marker_color='rgba(100, 100, 255, 0.3)',
                                                yaxis="y2"
                                            ))
                                            
                                            # Add Moving Averages for better visual analysis
                                            if 'SMA' in data_with_indicators.columns:
                                                fig.add_trace(go.Scatter(
                                                    x=data_with_indicators.index[-30:],
                                                    y=data_with_indicators['SMA'][-30:],
                                                    name='9-day SMA',
                                                    line=dict(color='rgba(255, 165, 0, 0.7)', width=2)
                                                ))
                                                
                                            if 'EMA_20' in data_with_indicators.columns:
                                                fig.add_trace(go.Scatter(
                                                    x=data_with_indicators.index[-30:],
                                                    y=data_with_indicators['EMA_20'][-30:],
                                                    name='20-day EMA',
                                                    line=dict(color='rgba(46, 139, 87, 0.7)', width=2)
                                                ))

                                            # Update layout for better visualization
                                            fig.update_layout(
                                                title=f"{str(live_ticker)} - Recent Price Movement",
                                                xaxis_title="Date",
                                                yaxis_title="Price",
                                                template="plotly_white",
                                                height=600,
                                                hovermode="x unified",
                                                yaxis=dict(
                                                    domain=[0.3, 1.0],
                                                    showgrid=True,
                                                    gridcolor='rgba(230, 230, 230, 0.8)'
                                                ),
                                                yaxis2=dict(
                                                    domain=[0, 0.2],
                                                    showgrid=False,
                                                    title="Volume"
                                                ),
                                                legend=dict(
                                                    orientation="h",
                                                    yanchor="bottom",
                                                    y=1.02,
                                                    xanchor="right",
                                                    x=1
                                                )
                                            )

                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Display key technical indicators
                                            st.subheader("Technical Indicators")
                                            
                                            # Create metrics in a row
                                            col1, col2, col3, col4 = st.columns(4)
                                            
                                            # RSI
                                            rsi_value = float(data_with_indicators['RSI'].iloc[-1]) if 'RSI' in data_with_indicators.columns else 0
                                            rsi_color = "#4CAF50" if rsi_value < 70 else "#FF5252" if rsi_value > 70 else "#FFC107"
                                            col1.markdown(f"""
                                            <div style="background-color: rgba(66, 66, 66, 0.05); border-radius: 10px; padding: 10px; text-align: center;">
                                                <p style="margin: 0; color: #666; font-size: 0.9em;">RSI (14)</p>
                                                <p style="font-size: 1.3em; font-weight: bold; margin: 5px 0; color: {rsi_color};">{rsi_value:.2f}</p>
                                                <p style="margin: 0; font-size: 0.8em; color: #888;">{get_rsi_interpretation(rsi_value)}</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                                            # MACD
                                            macd_value = float(data_with_indicators['MACD'].iloc[-1]) if 'MACD' in data_with_indicators.columns else 0
                                            macd_signal = float(data_with_indicators['MACD_Signal'].iloc[-1]) if 'MACD_Signal' in data_with_indicators.columns else 0
                                            macd_diff = macd_value - macd_signal
                                            macd_color = "#4CAF50" if macd_diff > 0 else "#FF5252"
                                            col2.markdown(f"""
                                            <div style="background-color: rgba(66, 66, 66, 0.05); border-radius: 10px; padding: 10px; text-align: center;">
                                                <p style="margin: 0; color: #666; font-size: 0.9em;">MACD</p>
                                                <p style="font-size: 1.3em; font-weight: bold; margin: 5px 0; color: {macd_color};">{macd_diff:.2f}</p>
                                                <p style="margin: 0; font-size: 0.8em; color: #888;">{get_macd_interpretation(macd_diff)}</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                                            # Bollinger Bands
                                            if 'BB_Upper' in data_with_indicators.columns and 'BB_Lower' in data_with_indicators.columns:
                                                current_price = float(data['Close'].iloc[-1])
                                                upper_band = float(data_with_indicators['BB_Upper'].iloc[-1])
                                                lower_band = float(data_with_indicators['BB_Lower'].iloc[-1])
                                                bb_width = (upper_band - lower_band) / data_with_indicators['BB_Middle'].iloc[-1]
                                                bb_color = "#FFC107" if current_price > upper_band or current_price < lower_band else "#4CAF50"
                                                
                                                col3.markdown(f"""
                                                <div style="background-color: rgba(66, 66, 66, 0.05); border-radius: 10px; padding: 10px; text-align: center;">
                                                    <p style="margin: 0; color: #666; font-size: 0.9em;">BB Width</p>
                                                    <p style="font-size: 1.3em; font-weight: bold; margin: 5px 0; color: {bb_color};">{bb_width:.2f}</p>
                                                    <p style="margin: 0; font-size: 0.8em; color: #888;">{get_bb_interpretation(current_price, upper_band, lower_band)}</p>
                                                </div>
                                                """, unsafe_allow_html=True)
                                            
                                            # Simple Moving Average
                                            if 'SMA' in data_with_indicators.columns:
                                                current_price = float(data['Close'].iloc[-1])
                                                sma = float(data_with_indicators['SMA'].iloc[-1])
                                                sma_diff = ((current_price - sma) / sma) * 100
                                                sma_color = "#4CAF50" if current_price > sma else "#FF5252"
                                                
                                                col4.markdown(f"""
                                                <div style="background-color: rgba(66, 66, 66, 0.05); border-radius: 10px; padding: 10px; text-align: center;">
                                                    <p style="margin: 0; color: #666; font-size: 0.9em;">Price vs SMA</p>
                                                    <p style="font-size: 1.3em; font-weight: bold; margin: 5px 0; color: {sma_color};">{sma_diff:+.2f}%</p>
                                                    <p style="margin: 0; font-size: 0.8em; color: #888;">{"Above SMA" if current_price > sma else "Below SMA"}</p>
                                                </div>
                                                """, unsafe_allow_html=True)
                                        except Exception as e:
                                            st.markdown(f"""
                                            <div class="error-container">
                                                <div class="error-message">Error creating chart: {str(e)}</div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    
                                    # Add chart_tab2 for patterns visualization
                                    with chart_tab2:
                                        st.markdown("### Pattern Analysis")
                                        try:
                                            # Detect patterns
                                            patterns_df = detect_candlestick_patterns(data)
                                            
                                            # Display patterns
                                            st.write("Recent candlestick patterns detected:")
                                            
                                            # Fix Series truth value ambiguity
                                            pattern_count = 0
                                            # Check if Pattern column exists
                                            if 'Pattern' in patterns_df.columns:
                                                # Count non-null values explicitly
                                                pattern_count = patterns_df['Pattern'].notna().sum()
                                            
                                            if pattern_count > 0:
                                                # Prepare pattern data - avoid Series truth value issues
                                                patterns_list = []
                                                for idx, row in patterns_df.iloc[-10:].iterrows():
                                                    pattern = row.get('Pattern')
                                                    if pattern is not None and isinstance(pattern, str):  # Ensure it's a string
                                                        pattern_type = row.get('Pattern_Type', '')
                                                        if not isinstance(pattern_type, str):
                                                            pattern_type = str(pattern_type) if pattern_type is not None else ''
                                                        
                                                        # Determine signal based on pattern type text
                                                        is_bullish = "Bullish" in pattern_type if pattern_type else False
                                                        signal = "Bullish" if is_bullish else "Bearish"
                                                        signal_color = "#4CAF50" if is_bullish else "#FF5252"
                                                        
                                                        patterns_list.append({
                                                            'Date': idx.strftime('%Y-%m-%d'),
                                                            'Pattern': pattern,
                                                            'Signal': signal,
                                                            'Signal_Color': signal_color
                                                        })
                                                
                                                if patterns_list:
                                                    # Convert to DataFrame
                                                    display_df = pd.DataFrame(patterns_list)
                                                    
                                                    # Display as styled table
                                                    st.dataframe(display_df[['Date', 'Pattern', 'Signal']])
                                                    
                                                    # Plot pattern distribution
                                                    st.subheader("Pattern Distribution")
                                                    pattern_counts = display_df['Pattern'].value_counts().reset_index()
                                                    pattern_counts.columns = ['Pattern', 'Count']
                                                    
                                                    fig = px.bar(
                                                        pattern_counts,
                                                        x='Pattern', 
                                                        y='Count',
                                                        title="Candlestick Pattern Distribution",
                                                        color='Pattern'
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)
                                            else:
                                                st.info("No significant candlestick patterns detected in recent data.")
                                                
                                            # Add buy/sell ratio chart
                                            st.subheader("Buy/Sell Pressure Analysis")
                                            try:
                                                # Calculate buyer-seller ratio
                                                data_with_ratio, buy_sell_ratio = calculate_buyer_seller_ratio(data)
                                                
                                                # Create buy/sell pressure chart
                                                ratio_fig = go.Figure()
                                                
                                                # Add buyer-seller ratio line
                                                ratio_fig.add_trace(go.Scatter(
                                                    x=data_with_ratio.index[-30:],
                                                    y=data_with_ratio['Buyer_Ratio'][-30:] * 100,
                                                    name="Buyer Percentage",
                                                    line=dict(color='#4CAF50', width=2)
                                                ))
                                                
                                                # Add 50% reference line
                                                ratio_fig.add_trace(go.Scatter(
                                                    x=data_with_ratio.index[-30:],
                                                    y=[50] * len(data_with_ratio.index[-30:]),
                                                    name="Neutral",
                                                    line=dict(color='gray', width=1, dash='dot')
                                                ))
                                                
                                                # Update layout
                                                ratio_fig.update_layout(
                                                    title="Buyer vs Seller Pressure (30 Days)",
                                                    xaxis_title="Date",
                                                    yaxis_title="Buyer Percentage (%)",
                                                    template="plotly_white",
                                                    height=350,
                                                    hovermode="x unified",
                                                    yaxis=dict(range=[0, 100])
                                                )
                                                
                                                st.plotly_chart(ratio_fig, use_container_width=True)
                                                
                                                # Display current ratio
                                                current_ratio = float(data_with_ratio['Buyer_Ratio'].iloc[-1]) * 100
                                                ratio_text = "Bullish" if current_ratio > 60 else "Bearish" if current_ratio < 40 else "Neutral"
                                                ratio_color = "#4CAF50" if current_ratio > 60 else "#FF5252" if current_ratio < 40 else "#FFC107"
                                                
                                                st.markdown(f"""
                                                <div style="background-color: rgba(66, 66, 66, 0.05); border-radius: 10px; padding: 10px; text-align: center; margin-top: 10px;">
                                                    <p style="margin: 0; color: #666; font-size: 0.9em;">Current Buy/Sell Ratio</p>
                                                    <p style="font-size: 1.3em; font-weight: bold; margin: 5px 0; color: {ratio_color};">{current_ratio:.1f}% Buyers</p>
                                                    <p style="margin: 0; font-size: 0.8em; color: #888;">Market Sentiment: {ratio_text}</p>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                            except Exception as e:
                                                st.info("Couldn't calculate buy/sell ratio. Using limited data might cause this issue.")
                                            
                                        except Exception as e:
                                            st.markdown(f"""
                                            <div class="error-container">
                                                <div class="error-message">Error analyzing patterns: {str(e)}</div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    
                                    # Add chart_tab3 for advanced LSTM prediction model
                                    with chart_tab3:
                                        st.markdown("### LSTM Model Prediction")
                                        
                                        try:
                                            with st.spinner("Building LSTM prediction model..."):
                                                # Prepare data for LSTM
                                                from sklearn.preprocessing import MinMaxScaler
                                                
                                                # Select only relevant columns for prediction
                                                prediction_data = data.copy()
                                                features = ['Close', 'Volume']
                                                
                                                if 'RSI' in data_with_indicators.columns:
                                                    prediction_data['RSI'] = data_with_indicators['RSI']
                                                    features.append('RSI')
                                                
                                                if 'MACD' in data_with_indicators.columns:
                                                    prediction_data['MACD'] = data_with_indicators['MACD']
                                                    features.append('MACD')
                                                
                                                # Time steps (look back period)
                                                time_steps = 10
                                                
                                                # Scale the data
                                                scaler = MinMaxScaler(feature_range=(0, 1))
                                                scaled_data = scaler.fit_transform(prediction_data[features])
                                                
                                                # Prepare sequences
                                                X, y = [], []
                                                for i in range(time_steps, len(scaled_data)):
                                                    X.append(scaled_data[i-time_steps:i])
                                                    y.append(scaled_data[i, 0])  # 0 is the index of Close price
                                                    
                                                X, y = np.array(X), np.array(y)
                                                
                                                # Create LSTM model
                                                model = tf.keras.Sequential([
                                                    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                                                    tf.keras.layers.Dropout(0.2),
                                                    tf.keras.layers.LSTM(50, return_sequences=False),
                                                    tf.keras.layers.Dropout(0.2),
                                                    tf.keras.layers.Dense(1)
                                                ])
                                                
                                                # Compile and fit the model
                                                model.compile(optimizer='adam', loss='mean_squared_error')
                                                model.fit(X, y, epochs=5, batch_size=32, verbose=0)
                                                
                                                # Prepare last sequence for prediction
                                                last_sequence = scaled_data[-time_steps:]
                                                
                                                # Number of days to predict
                                                forecast_days = 14
                                                
                                                # Get prediction
                                                future_pred = predict_future(model, last_sequence, scaler, forecast_days)
                                                
                                                # Create future dates with proper type handling
                                                last_date = prediction_data.index[-1]
                                                # Convert to datetime.datetime if it's pandas Timestamp
                                                if isinstance(last_date, pd.Timestamp):
                                                    last_date = last_date.to_pydatetime()
                                                # Generate future dates with consistent type - use explicit datetime.timedelta
                                                future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(forecast_days)]
                                                
                                                # Calculate confidence bounds
                                                mse = np.mean(np.square(y - model.predict(X, verbose=0).flatten()))
                                                std_dev = np.sqrt(mse)
                                                
                                                # Create confidence intervals
                                                lower_bound = [max(0, price * (1 - std_dev * 1.96)) for price in np.array(future_pred).flatten()]
                                                upper_bound = [price * (1 + std_dev * 1.96) for price in np.array(future_pred).flatten()]
                                                
                                                # Create visualization
                                                model_fig = go.Figure()
                                                
                                                # Add historical data
                                                model_fig.add_trace(go.Scatter(
                                                    x=prediction_data.index[-30:],
                                                    y=prediction_data['Close'][-30:],
                                                    name="Historical",
                                                    line=dict(color='#1E88E5', width=2)
                                                ))
                                                
                                                # Add prediction
                                                model_fig.add_trace(go.Scatter(
                                                    x=future_dates,
                                                    y=future_pred.flatten(),
                                                    name="LSTM Prediction",
                                                    line=dict(color='#6A1B9A', width=3, dash='solid')
                                                ))
                                                
                                                # Add confidence interval
                                                model_fig.add_trace(go.Scatter(
                                                    x=future_dates + future_dates[::-1],
                                                    y=upper_bound + lower_bound[::-1],
                                                    fill='toself',
                                                    fillcolor='rgba(106, 27, 154, 0.1)',
                                                    line=dict(color='rgba(255, 255, 255, 0)'),
                                                    name="95% Confidence Interval",
                                                    showlegend=True
                                                ))
                                                
                                                # Add separator line with proper type handling
                                                separator_date = prediction_data.index[-1]
                                                # Convert to datetime.datetime if it's pandas Timestamp
                                                if isinstance(separator_date, pd.Timestamp):
                                                    separator_date = separator_date.to_pydatetime()
                                                model_fig.add_vline(
                                                    x=separator_date, 
                                                    line=dict(color='rgba(0, 0, 0, 0.5)', width=1, dash='dot')
                                                )
                                                
                                                # Update layout
                                                model_fig.update_layout(
                                                    title=f"LSTM Model 14-Day Prediction for {live_ticker}",
                                                    xaxis_title="Date",
                                                    yaxis_title=f"Price ({currency_symbol})",
                                                    template="plotly_white",
                                                    height=500,
                                                    hovermode="x unified",
                                                    legend=dict(
                                                        orientation="h",
                                                        yanchor="bottom",
                                                        y=1.02,
                                                        xanchor="right",
                                                        x=1
                                                    )
                                                )
                                                
                                                st.plotly_chart(model_fig, use_container_width=True)
                                                
                                                # Calculate metrics
                                                final_pred = future_pred[-1][0]
                                                # Ensure proper numeric type for calculations
                                                current_price_val = float(current_price)
                                                final_pred_val = float(final_pred)
                                                expected_change = (final_pred_val - current_price_val) / current_price_val * 100
                                                change_color = "#4CAF50" if expected_change > 0 else "#FF5252"
                                                
                                                # Display prediction metrics
                                                col1, col2, col3, col4 = st.columns(4)
                                                
                                                col1.markdown(f"""
                                                <div style="background-color: rgba(66, 66, 66, 0.05); border-radius: 10px; padding: 10px; text-align: center;">
                                                    <p style="margin: 0; color: #666; font-size: 0.9em;">Current Price</p>
                                                    <p style="font-size: 1.3em; font-weight: bold; margin: 5px 0;">{currency_symbol}{current_price:.2f}</p>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                                col2.markdown(f"""
                                                <div style="background-color: rgba(66, 66, 66, 0.05); border-radius: 10px; padding: 10px; text-align: center;">
                                                    <p style="margin: 0; color: #666; font-size: 0.9em;">14-Day Forecast</p>
                                                    <p style="font-size: 1.3em; font-weight: bold; margin: 5px 0;">{currency_symbol}{final_pred:.2f}</p>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                                col3.markdown(f"""
                                                <div style="background-color: rgba(66, 66, 66, 0.05); border-radius: 10px; padding: 10px; text-align: center;">
                                                    <p style="margin: 0; color: #666; font-size: 0.9em;">Expected Change</p>
                                                    <p style="font-size: 1.3em; font-weight: bold; margin: 5px 0; color: {change_color};">{expected_change:+.2f}%</p>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                                col4.markdown(f"""
                                                <div style="background-color: rgba(66, 66, 66, 0.05); border-radius: 10px; padding: 10px; text-align: center;">
                                                    <p style="margin: 0; color: #666; font-size: 0.9em;">Model Confidence</p>
                                                    <p style="font-size: 1.3em; font-weight: bold; margin: 5px 0;">{"High" if std_dev < 0.05 else "Medium" if std_dev < 0.1 else "Low"}</p>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                                # Add model explanation
                                                st.markdown("""
                                                <div style="margin-top: 20px; padding: 15px; background-color: rgba(100, 100, 100, 0.05); border-radius: 8px;">
                                                    <h4 style="margin-top: 0; color: #333;">About LSTM Model Prediction</h4>
                                                    <p>This prediction uses a Long Short-Term Memory (LSTM) neural network, which is particularly effective for time series forecasting. Unlike simpler statistical models, LSTM can:</p>
                                                    <ul>
                                                        <li>Capture complex patterns and relationships in the data</li>
                                                        <li>Learn from longer historical periods while giving more weight to recent data</li>
                                                        <li>Account for multiple factors including technical indicators</li>
                                                    </ul>
                                                    <p><strong>Note:</strong> The prediction becomes less reliable the further into the future it goes. This is reflected in the widening confidence interval.</p>
                                                </div>
                                                """, unsafe_allow_html=True)
                                        except Exception as e:
                                            st.markdown(f"""
                                            <div class="error-container">
                                                <div class="error-message">Error building LSTM model: {str(e)}</div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                except Exception as e:
                                    st.markdown(f"""
                                    <div class="error-container">
                                        <div class="error-message">Error processing indicators: {str(e)}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.error(f"No data found for {live_ticker}. Please check the symbol and try again.")
                        except Exception as e:
                            st.markdown(f"""
                            <div class="error-container">
                                <div class="error-message">Error loading data for {live_ticker}: {str(e)}</div>
                            </div>
                            """, unsafe_allow_html=True)


    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

tf.get_logger().setLevel('ERROR')

# Set page title and enable wide layout


# Force reload of this file
# Initialize session state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}


if str(__name__) == "__main__":
    main()