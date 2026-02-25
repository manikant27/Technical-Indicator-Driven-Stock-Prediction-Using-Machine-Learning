# feature_engineering.py

import numpy as np
import pandas as pd

def add_technical_indicators(data):

    # Moving Averages
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()

    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = data['Close'].ewm(span=12).mean()
    ema26 = data['Close'].ewm(span=26).mean()
    data['MACD'] = ema12 - ema26

    # Bollinger Bands
    data['BB_upper'] = data['MA20'] + 2 * data['Close'].rolling(20).std()
    data['BB_lower'] = data['MA20'] - 2 * data['Close'].rolling(20).std()

    # MA Crossover
    data['MA_signal'] = np.where(data['MA20'] > data['MA50'], 1, 0)

    # Targets
    data['Target_reg'] = data['Close'].shift(-1)
    data['Target_class'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

    data.dropna(inplace=True)

    return data