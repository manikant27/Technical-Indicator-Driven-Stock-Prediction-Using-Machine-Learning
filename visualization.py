# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_price(data):
    plt.figure(figsize=(14,6))
    plt.plot(data['Close'], label='Close Price')
    plt.title("NIFTY 50 Price")
    plt.legend()
    plt.show()


def plot_indicators(data):
    plt.figure(figsize=(14,6))
    plt.plot(data['MA20'], label='MA20')
    plt.plot(data['MA50'], label='MA50')
    plt.title("Moving Averages")
    plt.legend()
    plt.show()


def plot_prediction(actual, predicted):
    plt.figure(figsize=(14,6))
    plt.plot(actual.values, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.title("Actual vs Predicted Price")
    plt.legend()
    plt.show()


def plot_correlation(data):
    plt.figure(figsize=(12,8))
    sns.heatmap(data.corr(), cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.show()