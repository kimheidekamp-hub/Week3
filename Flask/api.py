from flask import Flask
import requests
import pandas as pd 
import datetime

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

trained_model = None

def get_hourly_data(ticker, days = 100):
    # timestamp now
    timestamp = int(datetime.datetime.now().timestamp())
    # timestamp 100 days ago
    timestamp_100_days_ago = timestamp - days * 24 * 60 * 60

    url = f'https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?period1={timestamp_100_days_ago}&period2={timestamp}&interval=1h&includePrePost=true&events=div|split|earn&lang=en-US&region=US&source=cosaic'
    # add user agent to the header to avoid 403 error
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    data = response.json()
    return data

def hourly_data_to_dataframe(data):
    # create a dataframe with the timestamps and close indicator
    # get timestamps of the data
    timestamps = data['chart']['result'][0]['timestamp']
    close = data['chart']['result'][0]['indicators']['quote'][0]['close']
    volume = data['chart']['result'][0]['indicators']['quote'][0]['volume']
    low = data['chart']['result'][0]['indicators']['quote'][0]['low']
    high = data['chart']['result'][0]['indicators']['quote'][0]['high']
    open = data['chart']['result'][0]['indicators']['quote'][0]['open'] 

    df = pd.DataFrame({'timestamp': timestamps, 'close': close, 'volume': volume, 'low': low, 'high': high, 'open': open})
    # convert timestamp to datetime
    df['datetime'] = df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    # drop timestamp column
    df.drop('timestamp', axis=1, inplace=True)
    return df

def create_training_features(df):
    df_features = pd.DataFrame()
    for i in range(1, 3):
        df_features[f'close_{i}'] = df['close'].shift(i)
        df_features[f'volume_{i}'] = df['volume'].shift(i)
        df_features[f'low_{i}'] = df['low'].shift(i)
        df_features[f'high_{i}'] = df['high'].shift(i)
        df_features[f'open_{i}'] = df['open'].shift(i) 
    df_features['target'] = df['close']
    df_features['datetime'] = df['datetime']

    # drop rows with NaN values
    df_features.dropna(inplace=True)
    df_features = df_features.copy()
    dates = df_features['datetime']
    df_features.drop('datetime', axis=1, inplace=True)

    # print number of rows and columns in the dataframe
    print(f'Number of rows: {df_features.shape[0]}')
    print(f'Number of columns: {df_features.shape[1]}')
    return df_features, dates

def create_predict_features(df):
    # only use last row of the dataframe to create the features for the prediction
    
    df_features = pd.DataFrame()
    for i in range(1, 3):
        df_features[f'close_{i}'] = df['close'].shift(i-1)
        df_features[f'volume_{i}'] = df['volume'].shift(i-1)
        df_features[f'low_{i}'] = df['low'].shift(i-1)
        df_features[f'high_{i}'] = df['high'].shift(i-1)
        df_features[f'open_{i}'] = df['open'].shift(i-1)
    df_features['datetime'] = df['datetime']
    df_features = df_features.iloc[[-1]].copy()

    # drop rows with NaN values
    df_features.dropna(inplace=True)
    dates = df_features['datetime']
    df_features.drop('datetime', axis=1, inplace=True)

    # print number of rows and columns in the dataframe
    print(f'Number of rows: {df_features.shape[0]}')
    print(f'Number of columns: {df_features.shape[1]}')
    return df_features, dates

@app.route('/retrain')
def retrain():
    print('Retraining model...')
    df = hourly_data_to_dataframe(get_hourly_data('MSFT'))

    # code to retrain the model goes here
    df_features, dates = create_training_features(df)



    # split the data into train and test sets
    X = df_features.drop('target', axis=1)
    y = df_features['target']
    
    # create a linear regression model
    model = LinearRegression()
    # fit the model on the training data
    model.fit(X, y)
    global trained_model
    trained_model = model

    return 'Model retrained!'

@app.route('/predict')
def predict():
    if (trained_model is None):
        retrain()

    print('Making prediction...')
    # code to make a prediction goes here
    df = hourly_data_to_dataframe(get_hourly_data('MSFT', 1))

    df_features, dates = create_predict_features(df)
    
    # print the next whole hour after the last datetime in the dataframe,rounding to the next whole hour
    next_hour = dates.iloc[0] + datetime.timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
    print(f'Next whole hour: {next_hour}')
        
    y_pred = trained_model.predict(df_features)

    return {"target": y_pred.tolist()[0], "datetime": next_hour.strftime('%Y-%m-%d %H:%M:%S')}

@app.route('/')
def home():
    return 'Welcome to the API! Go to /retrain to retrain the model and /predict to make a prediction.'
