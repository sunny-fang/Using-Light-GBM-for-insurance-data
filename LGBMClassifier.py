#%%
# LBGMClassifier
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns 
import os
os.chdir('C:\\...')
#%%
data_train4 = pd.read_csv('train4.csv', encoding = 'Big5')
data_train4 = data_train4.drop(['Unnamed: 0'], axis = 1)
data_train3 = pd.read_csv('train3.csv', encoding = 'Big5')
train3_y1 = data_train3['Y1']
#train3_y1 = train3_y1.replace(['Y', 'N'], [1, 0])
train_data = pd.concat([data_train4, train3_y1], axis = 1)
#%%
target = train_data['Y1']
X = train_data.drop(['Y1'], axis = 1)
# Transform categorical features into the appropriate type that is expected by LightGBM
for c in X.columns:
    col_type = X[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        X[c] = X[c].astype('category')
X_train,X_test,y_train,y_test =train_test_split(X,target,test_size=0.2)
# gridsearch調參
# 參考網址 : https://juejin.im/post/5b76437ae51d45666b5d9b05
parameters = {
              'max_depth': [15, 20, 25, 30, 35],
              'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
              'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_freq': [2, 4, 5, 6, 8],
              'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
              'lambda_l2': [0, 10, 15, 35, 40],
              'cat_smooth': [1, 10, 15, 20, 35]
}
gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                         objective = 'binary',
                         metric = 'auc',
                         verbose = 0,
                         learning_rate = 0.01,
                         num_leaves = 35,
                         feature_fraction=0.8,
                         bagging_fraction= 0.9,
                         bagging_freq= 8,
                         lambda_l1= 0.6,
                         lambda_l2= 0)
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='accuracy', cv=3)
gsearch.fit(X_train, y_train)
print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))




















