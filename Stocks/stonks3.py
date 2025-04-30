from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)

def get_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data['Adj Close']

def plot_stock_data(stock_data, title):
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data.index, stock_data, label='Stock Price')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend()

    # Convert plot to bytes and embed in HTML
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode('utf8')
    plt.close()

    return img_data

def split_data(stock_data, test_size=0.2):
    train_data, test_data = train_test_split(stock_data, test_size=test_size, shuffle=False)
    return train_data, test_data

def train_linear_regression(train_data):
    X_train = np.array(range(len(train_data))).reshape(-1, 1)
    y_train = train_data.values
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(train_data, n_estimators=100):
    X_train = np.array(range(len(train_data))).reshape(-1, 1)
    y_train = train_data.values
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model

def generate_future_dates(end_date, days_to_predict):
    end_date = pd.to_datetime(end_date)  # Convert to datetime object
    future_dates = pd.date_range(start=end_date, periods=days_to_predict + 1, freq='B')[1:]
    return future_dates

def evaluate_model(model, test_data):
    X_test = np.array(range(len(test_data))).reshape(-1, 1)
    y_test = test_data.values
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse, predictions

def plot_predictions(stock_data, linear_reg_model, random_forest_model, future_dates):
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data.index, stock_data, label='Actual Price')

    # Linear Regression Predictions
    linear_reg_predictions = linear_reg_model.predict(np.array(range(len(stock_data), len(stock_data) + len(future_dates))).reshape(-1, 1))
    plt.plot(future_dates, linear_reg_predictions, label='Linear Regression Predictions', linestyle='dashed')

    # Random Forest Predictions
    random_forest_predictions = random_forest_model.predict(np.array(range(len(stock_data), len(stock_data) + len(future_dates))).reshape(-1, 1))
    plt.plot(future_dates, random_forest_predictions, label='Random Forest Predictions', linestyle='dashed')

    plt.title('Stock Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend()

    # Convert plot to bytes and embed in HTML
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode('utf8')
    plt.close()

    return img_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['stock_symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    days_to_predict = int(request.form['days_to_predict'])

    stock_data = get_stock_data(stock_symbol, start_date, end_date)
    historical_plot = plot_stock_data(stock_data, f'Historical Stock Prices for {stock_symbol}')

    train_data, test_data = split_data(stock_data)

    linear_reg_model = train_linear_regression(train_data)
    random_forest_model = train_random_forest(train_data)

    future_dates = generate_future_dates(end_date, days_to_predict)

    linear_reg_mse, _ = evaluate_model(linear_reg_model, test_data)
    random_forest_mse, _ = evaluate_model(random_forest_model, test_data)

    predictions_plot = plot_predictions(stock_data, linear_reg_model, random_forest_model, future_dates)

    return render_template('results.html', linear_reg_mse=linear_reg_mse, random_forest_mse=random_forest_mse,
                           historical_plot=historical_plot, predictions_plot=predictions_plot)

if __name__ == '__main__':
    app.run(debug=True)
