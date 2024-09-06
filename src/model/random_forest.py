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

rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42,max_depth=10)
rf_classifier.fit(X_train, y_train)
y_pred_tr = rf_classifier.predict(X_train)
y_pred = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred)
accuracy_rf_tr = accuracy_score(y_train, y_pred_tr)
f1_rf = f1_score(y_test, y_pred, average='micro')
f1_rf_tr = f1_score(y_train, y_pred_tr, average='micro')
print(f'Random Forest Training Accuracy: {accuracy_rf_tr:.2f}')
print(f'Random Forest Testing Accuracy: {accuracy_rf:.2f}')
print(f'Random Forest Training F1 score: {f1_rf_tr:.2f}')
print(f'Random Forest Testing F1 score: {f1_rf:.2f}')
