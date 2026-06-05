# %% [markdown]
# ## Voorspel aandeelprijs volgend uur
# 
# Setup

# %%
import requests
import pandas as pd 
import datetime

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# %% [markdown]
# ## Scrape Yahoo Finance voor historische data

# %%

# timestamp now
timestamp = int(datetime.datetime.now().timestamp())
# timestamp 100 days ago
timestamp_100_days_ago = timestamp - 100 * 24 * 60 * 60

def get_hourly_data(ticker):
    url = f'https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?period1={timestamp_100_days_ago}&period2={timestamp}&interval=1h&includePrePost=true&events=div|split|earn&lang=en-US&region=US&source=cosaic'
    # add user agent to the header to avoid 403 error
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    data = response.json()
    return data





# %%
#retrieve the data
data = get_hourly_data('MSFT')



# %%
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
# save data to a parquet file
df.to_parquet('data.parquet', index=False)


# %% [markdown]
# ## Laad de data uit de parquet file ipv via API

# %%
# load data from parquet file
df = pd.read_parquet('data.parquet')


# %%

# show the dataframe
df[21:50]

# %% [markdown]
# ## Maak simpel ML model
# 
# Voorspel op basis van de vorige drie uur wat de prijs wordt het komende uur. Zowel close, volume, low, high en open worden gebruikt. Simpele lineaire regressie.

# %%
# create a simple linear regression model to predict the close price based on the previous 24 hours

# create a new dataframe with the close price of the previous 24 hours as features and the close price of the current hour as target
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
#print(df_features)

# split the data into train and test sets
X = df_features.drop('target', axis=1)
y = df_features['target']
# do a train test split with the first 80% of the data in the train set and the last 20% of the data in the test set, not random
split = 0.8
X_train = X[:int(split*len(X))]
y_train = y[:int(split*len(y))]
X_test = X[int(split*len(X)):]
y_test = y[int(split*len(y)):]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Number of rows in train set: {X_train.shape[0]}')
print(f'Number of rows in test set: {X_test.shape[0]}')
# create a linear regression model
model = LinearRegression()
# fit the model on the training data
model.fit(X_train, y_train)
# evaluate the model on the test data
score = model.score(X_test, y_test)
print(f'R^2 score: {score}')    

# compute the root mean squared error of the model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
print(f'RMSE: {rmse}')

# create a dataframe with the actual and predicted values of the test set and the corresponding dates
chart_data = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
}, index=dates[int(split*len(X)):])

# plot the predicted values against the actual values over time with timestamps on the x-axis and close price on the y-axis, where the predicted value is red and the actual value is blue and format the x axis to show the date and time
plt.figure(figsize=(10,5))
plt.plot(chart_data.index, chart_data['Actual'], label='Actual', color='blue')
plt.plot(chart_data.index, chart_data['Predicted'], label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Price of MSFT')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=100))
plt.xticks(rotation=45)
plt.legend()
plt.show()  


# %%

# Nieuwste voorspelling printen met datum en tijd
print(chart_data['Predicted'].iloc[-1])
print(chart_data.index[-1])


# %% [markdown]
# ## Dashboard

# %%
import streamlit as st

# %%
st.line_chart(chart_data)


