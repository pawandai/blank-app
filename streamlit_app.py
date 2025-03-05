import os
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import model_from_json, Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from datetime import date
from keras.callbacks import EarlyStopping 
import glob

current_date=date.today()
# Download NLTK resources if not already downloaded
nltk.download('vader_lexicon')

# Define data sources for each company
COMPANIES = {
    'ADBL': {
        'price_url': 'https://raw.githubusercontent.com/pawandai/scrapers/refs/heads/master/data/company-wise/ADBL.csv',
        'headlines_url': 'https://raw.githubusercontent.com/pawandai/scrapers/refs/heads/master/data/news/raw_news.csv'
    },
    'AHPC': {
        'price_url': 'https://raw.githubusercontent.com/pawandai/scrapers/refs/heads/master/data/company-wise/AHPC.csv',
        'headlines_url': 'https://raw.githubusercontent.com/pawandai/scrapers/refs/heads/master/data/news/raw_news.csv'
    },
    'ALICL': {
        'price_url': 'https://raw.githubusercontent.com/pawandai/scrapers/refs/heads/master/data/company-wise/ALICL.csv',
        'headlines_url': 'https://raw.githubusercontent.com/pawandai/scrapers/refs/heads/master/data/news/raw_news.csv'
    },
    'BPCL': {
        'price_url': 'https://raw.githubusercontent.com/pawandai/scrapers/refs/heads/master/data/company-wise/BPCL.csv',
        'headlines_url': 'https://raw.githubusercontent.com/pawandai/scrapers/refs/heads/master/data/news/raw_news.csv'
    },
    'SBI': {
        'price_url': 'https://raw.githubusercontent.com/pawandai/scrapers/refs/heads/master/data/company-wise/SBI.csv',
        'headlines_url': 'https://raw.githubusercontent.com/pawandai/scrapers/refs/heads/master/data/news/raw_news.csv'
    },
}

def load_stock_data(price_url, headlines_url):
    """Load and merge stock price and headlines data."""
    stock_price = pd.read_csv(price_url)
    stock_price['published_date'] = pd.to_datetime(stock_price['published_date']).dt.normalize()
    stock_price.set_index('published_date', inplace=True)
    stock_price.sort_index(inplace=True)
    
    stock_headlines = pd.read_csv(headlines_url)
    stock_headlines['published_date'] = pd.to_datetime(stock_headlines['published_date'], errors='coerce').dt.normalize()
    stock_headlines = stock_headlines.groupby('published_date')['Title'].apply(lambda x: ','.join(x)).reset_index()
    stock_headlines.set_index('published_date', inplace=True)
    stock_headlines.sort_index(inplace=True)
    
    stock_data = pd.merge(stock_price, stock_headlines, left_index=True, right_index=True, how='inner')
    stock_data.dropna(inplace=True)
    return stock_data

def perform_sentiment(stock_data):
    """Perform sentiment analysis on headlines and add a 'compound' column."""
    sid = SentimentIntensityAnalyzer()
    stock_data['compound'] = stock_data['Title'].apply(lambda x: sid.polarity_scores(x)['compound'])
    stock_data.drop(['Title'], axis=1, inplace=True)
    return stock_data

def prepare_dataset(stock_data):
    """
    Select and order the necessary columns.
    Expected columns: 'close', 'open', 'high', 'low', 'traded_quantity', 'compound'
    """
    stock_data = stock_data[['close', 'compound', 'open', 'high', 'low', 'traded_quantity']]
    return stock_data

def create_shifted_data(stock_data):
    """
    Create a dataset where the target (close_price_shifted) is tomorrow's closing price.
    Also shifts the compound column for inclusion as a feature.
    """
    close_price = stock_data['close']
    compound = stock_data['compound']
    volume = stock_data['traded_quantity']
    open_price = stock_data['open']
    high = stock_data['high']
    low = stock_data['low']
    
    close_price_shifted = close_price.shift(-1)
    compound_shifted = compound.shift(-1)
    
    data = pd.concat([close_price, close_price_shifted, compound, compound_shifted, volume, open_price, high, low], axis=1)
    data.columns = ['close_price', 'close_price_shifted', 'compound', 'compound_shifted', 'volume', 'open_price', 'high', 'low']
    data.dropna(inplace=True)
    return data


#Clean old model files
def clean_old_model_files():
    """Delete old model files for all companies, keeping only today's models."""
    for company in COMPANIES.keys():
        # Define today's file names
        json_path = f"model_{company}_{current_date}.json"
        weights_path = f"model_{company}_{current_date}.weights.h5"
        
        # Find all model files related to this company
        model_files = glob.glob(f"model_{company}_*.json") + glob.glob(f"model_{company}_*.weights.h5")

        # Loop through and delete old files
        for file in model_files:
            if file not in [json_path, weights_path]:  # Skip today's files
                os.remove(file)
                   
        



def train_and_save_model(company):
    """Train an LSTM model for the given company and save the model files."""
    st.write(f"Training model for {company}...")
    urls = COMPANIES[company]
    stock_data = load_stock_data(urls['price_url'], urls['headlines_url'])
    stock_data = perform_sentiment(stock_data)
    stock_data = prepare_dataset(stock_data)
    data = create_shifted_data(stock_data)
    if len(data) < 10:
        st.error("Not enough data to train the model.")
        return None
    features = ['close_price', 'compound', 'compound_shifted', 'volume', 'open_price', 'high', 'low']
    x = data[features].values
    y = data['close_price_shifted'].values.reshape(-1, 1)
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y)
    
    # Reshape input for LSTM: (samples, timesteps, features)
    X = x_scaled.reshape(x_scaled.shape[0], x_scaled.shape[1], 1)
    
    model = Sequential()
    model.add(LSTM(120, return_sequences=True, activation='tanh', input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(120, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(120, activation='tanh'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    
    # Train the model (using fewer epochs for demo purposes)
    model.fit(X, y_scaled, epochs=30, batch_size=16,verbose=2,callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
    
    json_path = f"model_{company}_{current_date}.json"
    weights_path = f"model_{company}_{current_date}.weights.h5"

     # Remove old model files for the company
    for file in glob.glob(f"model_{company}_*.json") + glob.glob(f"model_{company}_*.weights.h5"):
        if file not in [json_path, weights_path]:
            os.remove(file)
    
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights_path)
    st.write(f"Model for {company} trained and saved.")

    clean_old_model_files();
    return model

def load_model(company):
    """Load pre-trained model for a company; if missing, train it first."""
    json_path = f"model_{company}_{current_date}.json"
    weights_path = f"model_{company}_{current_date}.weights.h5"
    if not os.path.exists(json_path) or not os.path.exists(weights_path):
        st.write(f"Model files for {company} not found. Initiating training...")
        model = train_and_save_model(company)
        return model
    with open(json_path, "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    model.compile(loss="mse", optimizer="adam", metrics=["mse", "mae"])
    return model

def compute_error_metrics(y_true, y_pred):
    """Compute error metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

def get_prediction(company):
    """
    Process the data for the selected company, load/train the model,
    predict tomorrow's closing price, and compute error metrics on a test set.
    Returns:
      - Tomorrow's predicted price,
      - Today's closing price,
      - Error metrics,
      - Processed data,
      - Test set dates,
      - Test set actual closing prices,
      - Test set predicted closing prices.
    """
    urls = COMPANIES[company]
    raw_data = load_stock_data(urls['price_url'], urls['headlines_url'])
    processed_data = perform_sentiment(raw_data.copy())
    processed_data = prepare_dataset(processed_data)
    data = create_shifted_data(processed_data)
    if data.empty:
        st.error("Not enough data to generate a prediction.")
        return None, None, None, None, None, None, None
    features = ['close_price', 'compound', 'compound_shifted', 'volume', 'open_price', 'high', 'low']
    x = data[features].values
    y = data['close_price_shifted'].values.reshape(-1, 1)
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y)
    
    # Split data into training and test sets (80/20 split)
    split_index = int(0.8 * len(x_scaled))
    X_test = x_scaled[split_index:]
    y_test = y_scaled[split_index:]
    test_dates = data.index[split_index:]
    
    model = load_model(company)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_pred_scaled = model.predict(X_test_reshaped)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_inv = scaler_y.inverse_transform(y_test)
    metrics_dict = compute_error_metrics(y_test_inv, y_pred)
    
    # Calculate accuracy percentage as (1 - MAPE) * 100
    accuracy = (1 - metrics_dict["MAPE"]) * 100
    metrics_dict["Accuracy (%)"] = accuracy
    
    # Predict tomorrow's closing price using the last record of the dataset
    last_feature = x_scaled[-1].reshape(1, x_scaled.shape[1], 1)
    pred_scaled = model.predict(last_feature)
    pred_price = scaler_y.inverse_transform(pred_scaled)[0][0]
    today_close = processed_data['close'].iloc[-1]
    
    return pred_price, today_close, metrics_dict, processed_data, test_dates, y_test_inv, y_pred

def plot_predicted_vs_actual(company, test_dates, actual, predicted):
    """
    Plot the predicted vs. actual closing prices over the test period as simple line curves.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test_dates, actual, label="Actual Close Price", color='blue')
    ax.plot(test_dates, predicted, label="Predicted Close Price", color='green')
    ax.set_title(f"{company} - Predicted vs Actual Close Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    return fig

# ------------------ Streamlit App UI ------------------

st.title("Stock Price Prediction App")
st.write("Select a company to view the predicted closing price for tomorrow.")

selected_company = st.selectbox("Select a Company", list(COMPANIES.keys()))

if st.button("Predict Tomorrow's Closing Price"):
    with st.spinner("Processing..."):
        prediction, today_close, metrics_dict, processed_data, test_dates, actual_test, predicted_test = get_prediction(selected_company)
        if prediction is not None:
            st.success(f"Predicted Tomorrow's Closing Price for {selected_company}: Rs {prediction:.2f}")
            st.write(f"Today's Closing Price: Rs {today_close:.2f}")
            st.subheader("Error Metrics on Test Set:")
            st.write(f"RMSE: {metrics_dict['RMSE']:.2f}")
            st.write(f"MAE: {metrics_dict['MAE']:.2f}")
            st.write(f"R2 Score: {metrics_dict['R2']:.2f}")
            st.write(f"MAPE: {metrics_dict['MAPE']:.2%}")
            
            
            # Plot predicted vs actual closing prices for the test set (simple curves)
            fig = plot_predicted_vs_actual(selected_company, test_dates, actual_test, predicted_test)
            st.pyplot(fig)
        else:
            st.error("Prediction could not be made.")