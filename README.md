# Stock Market Prediction App

This is a Streamlit-based stock market prediction application that uses Yahoo Finance data to fetch stock prices, perform technical analysis, and generate predictions.

## Features

- Real-time stock data from Yahoo Finance API
- Technical indicators (RSI, MACD, Stochastic, Bollinger Bands)
- Price predictions with confidence intervals
- Trading signals with rationale
- Support for both US and Indian markets

## How to Run the App

### Method 1: Using run_fixed_app.py (Recommended)

This method will automatically install all required dependencies and run the app:

```bash
python run_fixed_app.py
```

### Method 2: Manual Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run fix_app.py
```

## Technical Details

The app uses:

- **yfinance** for reliable stock data fetching
- **Pandas** and **NumPy** for data manipulation
- **Plotly** for interactive charts
- **TensorFlow** for machine learning models
- **Streamlit** for the web interface

## Troubleshooting

If you encounter issues:

1. Make sure you have Python 3.8+ installed
2. Verify internet connectivity for stock data fetching
3. Check that all dependencies are installed correctly
4. Ensure stock symbols are entered correctly (US stocks: AAPL, MSFT; Indian stocks: RELIANCE.NS, TCS.NS)

## Notes

- Predictions are based on historical data and should not be used as the sole basis for investment decisions
- Technical indicators are provided for educational purposes
- Always do your own research before making investment decisions
