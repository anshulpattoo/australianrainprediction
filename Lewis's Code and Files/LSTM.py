#%% LSTM Second try
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

#%% Loading Data
dt = pd.read_csv("weatherAUS.csv")

summaryMath = dt.describe()
summaryAll = dt.describe(include='all')
print(dt.isna().sum())

#%% Cleaning Data

# Remove Cloumns: Cloud 9am Cloud 3pm Evaporation Sunshine. Too many missing values
dt = dt.drop(columns=['Cloud9am','Cloud3pm', 'Evaporation', 'Sunshine'])

# Remove Rows of few na
dt = dt.dropna(subset=['MinTemp','MaxTemp','Rainfall','WindDir3pm','WindSpeed9am',
                      'WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm',
                      'Temp9am','Temp3pm','RainToday','RainTomorrow'])

print(dt.isna().sum())

# NA Values  overlap completely so it is worth keeping for both columns
dt = dt.dropna(subset=['WindGustDir'])


dt = dt.dropna(subset=['Pressure9am','Pressure3pm'])

# Already have more recent data so not really worth keeping
dt = dt.drop(columns=['WindDir9am'])
print(dt.isna().sum()) 

# Data is clean of missing values

#%% Histogram
dt.hist(bins=30, figsize=(15,10))

#%% Cleaning String values
dt['Date'] = dt['Date'].astype('string')

dt['Year'] = dt['Date'].str.slice(stop=4).astype(int) - 2007
dt['Month'] = dt['Date'].str.slice(start=6, stop=7)
dt['Day'] = dt['Date'].str.slice(start=8)

# Not including day column because there aren't samples for that level
# of granularity to be useful

dt = dt.drop(columns=['Date'])

#TESTING LINE BELOW
Albury = dt[dt['Location'] == "Albury"]
#

dt = pd.get_dummies(data=dt, columns=['Location', 'WindGustDir','WindDir3pm'])
#dt = pd.get_dummies(data=dt, columns=['Year','Month', 'Location', 'WindGustDir','WindDir3pm'])

dt['RainToday'].astype('string')
dt['RainTomorrow'].astype('string')

dt['RainToday'] = dt['RainToday'].map({'Yes': 1, 'No': 0})
dt['RainTomorrow'] = dt['RainTomorrow'].map({'Yes': 1, 'No': 0})

#%% PLOTTING


values = Albury.values
# specify columns to plot
groups = [1,2,3,5,7,8,9,10,11,12,13,14]
i = 1
# plot each column
plt.figure(figsize = (100,50))
for group in groups:
	plt.subplot(len(groups), 1, i)
	plt.plot(values[:, group])
	plt.title(Albury.columns[group], y=0.5, loc='right')
	i += 1
plt.show()

#%%
from sklearn.model_selection import train_test_split

X = dt.drop(columns='RainTomorrow')
y = dt['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

## NORMALIZATION INCREASES ACCURACY +1%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

