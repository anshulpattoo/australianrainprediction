
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

dt['Year'] = dt['Date'].str.slice(stop=4).astype(int) - 2007
dt['Month'] = dt['Date'].str.slice(start=6, stop=7)
dt['Day'] = dt['Date'].str.slice(start=8)

# Not including day column because there aren't samples for that level
# of granularity to be useful

dt = dt.drop(columns=['Date'])

dt = pd.get_dummies(data=dt, columns=['Location', 'WindGustDir','WindDir3pm'])
#dt = pd.get_dummies(data=dt, columns=['Year','Month', 'Location', 'WindGustDir','WindDir3pm'])

dt['RainToday'].astype('string')
dt['RainTomorrow'].astype('string')

dt['RainToday'] = dt['RainToday'].map({'Yes': 1, 'No': 0})
dt['RainTomorrow'] = dt['RainTomorrow'].map({'Yes': 1, 'No': 0})

dt['Last_days'] = dt['RainTomorrow'].rolling(15).sum().shift(1)

dt['Last_days'] = dt['Last_days'].fillna(0)
#%%
dt.to_csv("lewis_preprocessed.csv")
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
#%%
model = Sequential()

model.add(Dense(93,input_shape=(93,), activation='relu'))
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

tracker = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test,y_test))

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
X_train_res, y_train_res = sm.fit_resample(X_train, y_train, k_neighbors = 10)

model2 = Sequential()

model2.add(Dense(93,input_shape=(93,),activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(1, activation='sigmoid'))


model2.compile(optimizer="adam",
              loss="binary_crossentropy", 
              metrics=["accuracy"])

tracker = model2.fit(X_train_res, y_train_res, epochs=6, batch_size=32, validation_data=(X_test,y_test))

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

y_pred = model2.predict(X_test)

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

#%%

from imblearn.over_sampling import ADASYN

sm = ADASYN(n_neighbors = 5, sampling_strategy='minority')

X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


model2 = Sequential()

model2.add(Dense(93,input_shape=(93,),activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(1, activation='sigmoid'))


model2.compile(optimizer="adam",
              loss="binary_crossentropy", 
              metrics=["accuracy"])

tracker = model2.fit(X_train_res, y_train_res, epochs=6, batch_size=32, validation_data=(X_test,y_test))

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

y_pred = model2.predict(X_test)

y_pred = np.rint(y_pred)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print(cm)

#%% LSTM Second try

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
def show_heatmap(data):
    plt.figure(figsize = (100,50))
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()
    

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(dt)
data = scaler.transform(dt)
corr = pd.DataFrame(data[:, 0:14]).corr()
show_heatmap(pd.DataFrame(data[:, 0:14]))


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
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import LSTM, Input
step = 1

past = 4
future = 0
learning_rate = 0.01
batch_size = 256
epochs = 3

start = past + future
#end = start + train_split


sequence_length = int(past / step)

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    X_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    X_test,
    y_test,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

for batch in dataset_train.take(1):
    inputs, targets = batch
    
model = Sequential()
model.add(Input(shape=(inputs.shape[1], inputs.shape[2])))
model.add(LSTM(32))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer="adam",
              loss="binary_crossentropy")
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test,y_test))
y_pred = model.predict(X_test)
y_pred=np.rint(y_pred)

#i = 0
#for k in y_pred:
#    if k>0:
#        y_pred[i] = 1
#    else:
#        y_pred[i] = 0
        



model.summary()
'''
path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


visualize_loss(history, "Training and Validation Loss")'''
#%%
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print(cm)
