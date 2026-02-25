# data_loader.py

import yfinance as yf
import pandas as pd

def load_data(ticker="^NSEI", start="2012-01-01", end="2024-12-31"):
    
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False
    )

    # Fix multi-index columns issue
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[['Open','High','Low','Close','Volume']]
    data.dropna(inplace=True)

    return data