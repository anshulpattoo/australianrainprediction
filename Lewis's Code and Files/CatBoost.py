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
