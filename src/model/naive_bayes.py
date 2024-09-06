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

naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)
y_pred_tr = naive_bayes_model.predict(X_train)
y_pred = naive_bayes_model.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred)
accuracy_nb_tr = accuracy_score(y_train, y_pred_tr)
f1_nb = f1_score(y_test, y_pred, average='micro')
f1_nb_tr = f1_score(y_train, y_pred_tr, average='micro')
print(f'Naive Bayes Training Accuracy: {accuracy_nb_tr:.2f}')
print(f'Naive Bayes Testing Accuracy: {accuracy_nb:.2f}')
print(f'Naive Bayes Training F1 score: {f1_nb_tr:.2f}')
print(f'Naive Bayes Testing F1 score: {f1_nb:.2f}')