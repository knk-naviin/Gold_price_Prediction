import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        st.stop()

model = load_pickle('gold_price_model.pkl')
scaler = load_pickle('scaler.pkl')

# Streamlit app
st.title('Gold Price Prediction')
st.write('Enter a year and month to predict the gold price (INR per gram) and get a buy recommendation.')

# Input fields
year = st.number_input('Year (e.g., 2025)', min_value=2010, max_value=2050, step=1)
month = st.number_input('Month (1-12)', min_value=1, max_value=12, step=1)

# Features
features = ['Open', 'High', 'Low', 'Vol.', 'Year', 'Month', 'Moving_Avg_20', 
            'Volatility', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Change %']

# Predict button
if st.button('Predict Gold Price'):
    try:
        # Load dataset
        df = pd.read_csv('Gold_Futures_Historical_Data.csv')

        # Preprocess data
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
        df = df.dropna(subset=['Date']).reset_index(drop=True)

        # Rename and clean columns
        df.rename(columns={'Price': 'Close'}, inplace=True)
        df['Vol.'] = df['Vol.'].apply(lambda x: float(x.replace(',', '').replace('K', '')) * 1000 
                                       if isinstance(x, str) and 'K' in x else 
                                       float(x.replace(',', '')) if isinstance(x, str) else x)
        df['Change %'] = df['Change %'].str.replace('%', '').astype(float)
        for col in ['Close', 'Open', 'High', 'Low']:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)

        # Generate features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Moving_Avg_20'] = df['Close'].rolling(window=20).mean().fillna(df['Close'].mean())
        df['Volatility'] = df['Close'].rolling(window=20).std().fillna(df['Close'].std())
        for lag in range(1, 4):
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        # Prepare input data
        month_data = df[df['Month'] == month]
        input_data = {}
        for feature in features:
            if feature == 'Year':
                input_data[feature] = year
            elif feature == 'Month':
                input_data[feature] = month
            elif feature in month_data.columns and len(month_data) > 0:
                input_data[feature] = month_data[feature].mean()
            else:
                input_data[feature] = df[feature].mean()

        # Debug: Show input data
        st.write("Input Features (Before Scaling):", input_data)

        input_df = pd.DataFrame([input_data], columns=features)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        st.write(f'Predicted Gold Price for {month}/{year}: ₹{prediction:.2f} per gram')

        # Recommendation
        input_date = pd.to_datetime(f'{year}-{month}-01')
        historical_mean = df['Close'].mean()
        recent_data = df[df['Date'] < input_date].tail(180)  # Last 6 months
        rolling_mean = recent_data['Close'].mean() if not recent_data.empty else historical_mean

        if prediction < historical_mean:
            st.success(f'Recommendation: BUY (Price is below historical mean of ₹{historical_mean:.2f})')
        elif prediction < rolling_mean:
            st.success(f'Recommendation: BUY (Price is below 6-month mean of ₹{rolling_mean:.2f})')
        else:
            st.warning(f'Recommendation: HOLD (Price is above historical mean of ₹{historical_mean:.2f} and 6-month mean of ₹{rolling_mean:.2f})')

        # Plot historical prices
        st.subheader('Historical Gold Prices')
        st.line_chart(df.set_index('Date')['Close'])

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
