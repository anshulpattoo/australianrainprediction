
#%% Importing Libraries
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

dt['Year'] = dt['Date'].str.slice(stop=4)
dt['Month'] = dt['Date'].str.slice(start=6, stop=7)

################## ADD DAYS TOO


# Not including day column because there aren't samples for that level
# of granularity to be useful

dt = dt.drop(columns=['Date'])

dt = pd.get_dummies(data=dt, columns=['Year','Month', 'Location', 'WindGustDir','WindDir3pm'])

dt['RainToday'].astype('string')
dt['RainTomorrow'].astype('string')

dt['RainToday'] = dt['RainToday'].map({'Yes': 1, 'No': 0})
dt['RainTomorrow'] = dt['RainTomorrow'].map({'Yes': 1, 'No': 0})

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
#%%
input_shape = (111)

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import InputLayer
from keras.optimizers import Adam

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

model = Sequential()

model.add(Dense(110,input_shape=(110,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer="adam",
              loss="binary_crossentropy", 
              metrics=["accuracy"])

tracker = model.fit(X_train, y_train, epochs=6, batch_size=32, validation_data=(X_test,y_test))

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(tracker.history["accuracy"], label = "training_accuracy")
ax.plot(tracker.history["val_accuracy"], label = "val_accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
ax.legend(loc = 'best', shadow = True,)
plt.show()

fig, ax = plt.subplots(figsize = (8,6))
ax.plot(tracker.history["loss"], label = "training_loss")
ax.plot(tracker.history["val_loss"], label = "val_loss")
plt.xlabel("epochs")
plt.ylabel("loss function")
ax.legend(loc = 'upper center', shadow = True,)
plt.show()

y_pred = model.predict(X_test)

y_pred = np.rint(y_pred)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print(cm)

########## RESULTS ##########

###############################
## NO DROPOUT
##
#               precision    recall  f1-score   support

#           0       0.88      0.95      0.92     27893
#           1       0.77      0.55      0.65      7808

#    accuracy                           0.87     35701
#   macro avg       0.83      0.75      0.78     35701
#weighted avg       0.86      0.87      0.86     35701

#                  PREDICTED
#                 0      1
# ACTUAL      0 [26612  1281]
#             1 [ 3479  4329]

###############################
## DROPOUT .2
##
#               precision    recall  f1-score   support

#           0       0.89      0.95      0.92     27893
 #          1       0.75      0.58      0.65      7808 

 #   accuracy                           0.87     35701
 #  macro avg       0.82      0.76      0.79     35701
#weighted avg       0.86      0.87      0.86     35701

#                  PREDICTED
#                 0      1
# ACTUAL      0 [26408  1485]
#             1 [ 3288  4520]

###############################
## DROPOUT .3
##
#               precision    recall  f1-score   support
#
#           0       0.87      0.97      0.91     27792
#           1       0.82      0.47      0.60      7909

#    accuracy                           0.86     35701
#   macro avg       0.84      0.72      0.76     35701
#weighted avg       0.85      0.86      0.84     35701

#                  PREDICTED
#                 0      1
# ACTUAL      0 [26946   846]
#             1 [ 4164  3745]

###############################

#%% Synthetic Minority Oversampling Technique (SMOTE)

from imblearn.over_sampling import SMOTE

sm = SMOTE()
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

model2 = Sequential()

model2.add(Dense(110,input_shape=(110,), activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(1, activation='softmax'))


model2.compile(optimizer="adam",
              loss="binary_crossentropy", 
              metrics=["accuracy"])

tracker = model.fit(X_train_res, y_train_res, epochs=6, batch_size=32, validation_data=(X_test,y_test))

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(tracker.history["accuracy"], label = "training_accuracy")
ax.plot(tracker.history["val_accuracy"], label = "val_accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
ax.legend(loc = 'best', shadow = True,)
plt.show()

fig, ax = plt.subplots(figsize = (8,6))
ax.plot(tracker.history["loss"], label = "training_loss")
ax.plot(tracker.history["val_loss"], label = "val_loss")
plt.xlabel("epochs")
plt.ylabel("loss function")
ax.legend(loc = 'upper center', shadow = True,)
plt.show()

y_pred = model.predict(X_test)

y_pred = np.rint(y_pred)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print(cm)

########## RESULTS ##########

###############################
#              precision    recall  f1-score   support

#           0       0.93      0.84      0.88     27904
#           1       0.57      0.77      0.66      7797

#    accuracy                           0.82     35701
#   macro avg       0.75      0.80      0.77     35701
#weighted avg       0.85      0.82      0.83     35701

#                  PREDICTED
#                 0      1
# ACTUAL      0 [23414  4490]
#             1 [ 1796  6001]

#  https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
# Worked Surpisingly well. Will be more likely to give a false positive than a false negative.
# Still not great on predicting positve class.

#%% CATBOOST
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, f1_score
from catboost import CatBoostClassifier
from tqdm import tqdm
from IPython.display import clear_output
from tensorflow import keras
from keras.layers import Dense, LSTM, Dropout
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)

def validate(model, val_data):
    y = model.predict(val_data[0])
    print('Accuracy =', accuracy_score(y, val_data[1]))
    print('ROC AUC =', roc_auc_score(y, val_data[1]))
    print('F1 =', f1_score(y, val_data[1]))
    
orig_data = pd.read_csv('weatherAUS.csv')
data = orig_data.copy()

cat, num = [], [] # find categorical and float columns
for col in data.drop(columns=['Date', 'RainTomorrow']).columns:
    if data[col].dtype == np.number:
        num.append(col)
    else:
        cat.append(col)
        
data.dropna(inplace=True)

data['Date'] = pd.to_datetime(data['Date'])
day, month = np.array([], dtype='int8'), np.array([], dtype='int8')
with tqdm(total=len(data)) as pb:
    for index, val in data['Date'].iteritems():
        day = np.append(day, val.day)
        month = np.append(month, val.month)
        pb.update(1)
data.insert(0, 'Day', day)
data.insert(0, 'Month', month)
data.drop(columns='Date', inplace=True)
cat += ['Day', 'Month']

# One hot encoding
data = pd.get_dummies(data, columns=cat)

data['RainTomorrow'] = data['RainTomorrow'].astype('category').cat.codes

data.insert(0, 'Last_days', data['RainTomorrow'].rolling(15).sum().shift(1))
data = data[15:]

X = data.drop(columns='RainTomorrow')
y = data['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=45)
val_data = (X_test, y_test)
cat_f = [] # Categorical columns for catboost
for col in X.columns:
    if X[col].dtype == np.uint8:
        cat_f.append(col)
        

model = CatBoostClassifier()
model.fit(X_train, y_train, verbose=0, cat_features=cat_f)
validate(model, val_data)

model_tun = CatBoostClassifier()
grid = {'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]}
model_tun.randomized_search(grid, X=X, y=y)
clear_output()

validate(model_tun, val_data)

#%%
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model_tun.predict(X_test)
y_pred = np.rint(y_pred)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print(cm)
#%%
model_final = CatBoostClassifier(learning_rate=0.3,depth=10, l2_leaf_reg=9)
model_final.fit(X_train, y_train, verbose=0, cat_features=cat_f)

from sklearn.metrics import classification_report, confusion_matrix
y_pred = model_final.predict(X_test)
y_pred = np.rint(y_pred)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print(cm)

#              precision    recall  f1-score   support
#
 #          0       0.90      0.94      0.92     11031
  #         1       0.74      0.61      0.67      3071
#
 #   accuracy                           0.87     14102
  # macro avg       0.82      0.77      0.79     14102
#weighted avg       0.86      0.87      0.86     14102

#[[10389   642]
 #[ 1207  1864]]

#%%
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model_final, X = X_train, y = y_train, cv = 10)
print("Accuracy:{:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation:{:.2f} %".format(accuracies.std()*100))

#Accuracy:86.54 %
#Standard Deviation:0.47 %

#%% LSTM