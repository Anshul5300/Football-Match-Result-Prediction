import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.tree import export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle



data = "../data/PremierProcessed/mergeDataOriginal.csv"

X = data.drop('FTR',axis = 1)
y= data['FTR']

dt_classifier = DecisionTreeClassifier(random_state=42,max_features='sqrt')
dt_classifier.fit(X_train, y_train)
y_pred_tr = dt_classifier.predict(X_train)
y_pred = dt_classifier.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred)
accuracy_dt_tr = accuracy_score(y_train, y_pred_tr)
f1_dt = f1_score(y_test, y_pred, average='micro')
f1_dt_tr = f1_score(y_train, y_pred_tr, average='micro')
print(f'Decision Tree Training Accuracy: {accuracy_dt_tr:.2f}')
print(f'Decision Tree Testing Accuracy: {accuracy_dt:.2f}')
print(f'Decision Tree Training F1 score: {f1_dt_tr:.2f}')
print(f'Decision Tree Testing F1 score: {f1_dt:.2f}')