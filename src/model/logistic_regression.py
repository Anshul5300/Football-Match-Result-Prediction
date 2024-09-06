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

logistic_regression_model = LogisticRegression(random_state=42,multi_class='ovr',max_iter=500)
logistic_regression_model.fit(X_train, y_train)
y_pred_tr = naive_bayes_model.predict(X_train)
y_pred = logistic_regression_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred)
accuracy_lr_tr = accuracy_score(y_train, y_pred_tr)
f1_lr = f1_score(y_test, y_pred, average='micro')
f1_lr_tr = f1_score(y_train, y_pred_tr, average='micro')
print(f'Logistic Regression Training Accuracy: {accuracy_lr_tr:.2f}')
print(f'Logistic Regression Testing Accuracy: {accuracy_lr:.2f}')
print(f'Logistic Regression Training F1 score: {f1_lr_tr:.2f}')
print(f'Logistic Regression Testing F1 score: {f1_lr:.2f}')