import os
import pandas as pd
import yfinance as yf
import streamlit as st
import re
import datetime
import traceback

def clean_ticker(ticker):
    """Clean ticker symbol to handle various input formats"""
    if isinstance(ticker, (list, tuple)):
        ticker = ''.join([str(x) for x in ticker])
    elif isinstance(ticker, str) and '(' in ticker and ')' in ticker:
        matches = re.findall(r"['\"](w)['\"]+|\w+", ticker)
        if matches:
            ticker = ''.join(matches)
    
    ticker = re.sub(r'[^a-zA-Z0-9\.]', '', str(ticker))
    return ticker.upper()

def load_stock_data(ticker, start_date=None, end_date=None):
    """Load stock data using yfinance"""
    if not ticker:
        st.error("No ticker symbol provided")
        return None

    cleaned_ticker = clean_ticker(ticker)
    tickers_to_try = [cleaned_ticker]
    
    if '.' not in cleaned_ticker:
        tickers_to_try.append(f"{cleaned_ticker}.US")
    elif cleaned_ticker.endswith('.NS'):
        tickers_to_try.append(cleaned_ticker.rsplit('.', 1)[0])
    
    st.write("Attempting to fetch data for tickers:", tickers_to_try)

    # Handle dates
    current_date = datetime.datetime.now(datetime.timezone.utc).date()
    if start_date is None:
        start_date = (pd.Timestamp.now() - pd.DateOffset(years=3)).date()
    else:
        try:
            start_date = pd.to_datetime(start_date).date()
        except Exception:
            start_date = (pd.Timestamp.now() - pd.DateOffset(years=3)).date()
    
    if end_date is None:
        end_date = current_date
    else:
        try:
            end_date = pd.to_datetime(end_date).date()
            if end_date > current_date:
                end_date = current_date
        except Exception:
            end_date = current_date

    # Check market hours
    current_time_et = datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=-5)))
    is_market_open = (
        current_time_et.weekday() < 5 and
        current_time_et.replace(hour=9, minute=30) <= current_time_et <= current_time_et.replace(hour=16)
    )

    # Adjust end date if market is open
    download_end_date = (current_date - datetime.timedelta(days=1)) if is_market_open else end_date

    # Try yfinance with different ticker variants
    for ticker_variant in tickers_to_try:
        try:
            # Format dates explicitly as strings
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = (download_end_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            
            st.write(f"Fetching {ticker_variant} data from {start_date_str} to {end_date_str}")
            
            data = yf.download(
                ticker_variant,
                start=start_date_str,
                end=end_date_str,
                progress=False,
                auto_adjust=False  # Explicitly set auto_adjust to False for consistent data
            )
            
            # Debug information
            st.write(f"Data shape for {ticker_variant}:", data.shape)
            if not data.empty:
                st.write(f"First few rows for {ticker_variant}:")
                st.write(data.head())
                
                # Ensure consistent timezone handling
                data.index = data.index.tz_localize(None)
                # Convert download_end_date to pandas Timestamp for comparison
                end_date_ts = pd.Timestamp(download_end_date).normalize()
                # Filter data using proper datetime comparison
                data = data[data.index.normalize() <= end_date_ts]
                if not data.empty:
                    return data
            else:
                st.write(f"No data returned for {ticker_variant}")
        except Exception as e:
            st.error(f"Error fetching {ticker_variant}:")
            st.error(f"Exception: {str(e)}")
            st.error("Full traceback:")
            st.code(traceback.format_exc())
            continue

    st.error(f"Could not fetch data for {cleaned_ticker}")
    return None