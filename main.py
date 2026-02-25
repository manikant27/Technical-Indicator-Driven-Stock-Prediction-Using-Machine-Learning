# main.py

from data_loader import load_data
from feature_engineering import add_technical_indicators
from model_training import train_models
from visualization import plot_price, plot_indicators, plot_prediction, plot_correlation


# Load Data
data = load_data()

# Add Features
data = add_technical_indicators(data)

# Visualize Raw Data
plot_price(data)
plot_indicators(data)
plot_correlation(data)

# Train Models
rmse, acc, regimes, reg_pred, y_test = train_models(data)

print("Regression RMSE:", rmse)
print("Classification Accuracy:", acc)

# Plot Predictions
plot_prediction(y_test, reg_pred)