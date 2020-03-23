#%%
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
#%%
os.chdir('D:\\研究所(2019.9.23)\\碩一上\\國泰大數據競賽\\刪變數後資料集')
X = pd.read_csv('train6.csv', encoding = 'Big5')
X['Y1'] = X['Y1'].replace(['Y', 'N'], [1, 0])
for c in X.columns:
    col_type = X[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        X[c] = X[c].astype('category')
x_categorical = X.dtypes == 'category'
x_categorical = DataFrame(x_categorical)
aa = x_categorical[x_categorical[0] == True]
type(aa)
categorical_var = list(aa.index)
target = X['Y1']
X = X.drop(['Y1'], axis = 1)

X_train,X_test,y_train,y_test =train_test_split(X,target,test_size=0.2, random_state=0)

params = {    
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'nthread':4,
          'learning_rate':0.005,
          'num_leaves':16, 
          'max_depth': 4,   
          'subsample': 0.8, 
          'colsample_bytree': 0.8, 
          'max_bin':15,
          'min_data_in_leaf':71,
          'bagging_fraction':0.6,'bagging_freq': 0, 'feature_fraction': 1,
          'lambda_l1':0.3, 'lambda_l2':0.5
    }

data_train = lgb.Dataset(X_train, y_train, categorical_feature = categorical_var)
cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',early_stopping_rounds=50,seed=0)
print('best n_estimators:', len(cv_results['auc-mean']))    # n = 52，調參後變 n = 1000
print('best cv score:', pd.Series(cv_results['auc-mean']).max())
#%%
# 開始調參
from sklearn.model_selection import GridSearchCV
params_test1={'max_depth': range(3,8,1), 'num_leaves':range(5, 100, 5)}    
params_test1={'max_depth': [4], 'num_leaves':range(15, 21, 1)} 
gsearch1 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=52), 
                       param_grid = params_test1, scoring='roc_auc',cv=10,n_jobs=-1)
gsearch1.fit(X_train,y_train)
gsearch1.best_params_, gsearch1.best_score_    # 'max_depth': 4, 'num_leaves': 16

params_test2={'max_bin': range(5,256,10), 'min_data_in_leaf':range(1,102,10)}
gsearch2 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=52, max_depth=4, num_leaves=16), 
                       param_grid = params_test2, scoring='roc_auc',cv=10,n_jobs=-1)
gsearch2.fit(X_train,y_train)
gsearch2.best_params_, gsearch1.best_score_    # 'max_bin': 15, 'min_data_in_leaf': 71

params_test3={'feature_fraction': [0.6,0.7,0.8,0.9,1.0],
              'bagging_fraction': [0.6,0.7,0.8,0.9,1.0],
              'bagging_freq': range(0,81,10)
}
gsearch3 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=52, max_depth=4, num_leaves=16, max_bin=15, min_data_in_leaf=71), 
                       param_grid = params_test3, scoring='roc_auc',cv=10,n_jobs=-1)
gsearch3.fit(X_train,y_train)
gsearch3.best_params_, gsearch3.best_score_    # 'bagging_fraction': 0.6, 'bagging_freq': 0, 'feature_fraction': 1

params_test4={'lambda_l1': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0],
              'lambda_l2': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]
}     
gsearch4 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=52, max_depth=4, num_leaves=16,max_bin=15,min_data_in_leaf=71, 
                                                       bagging_fraction=0.6,bagging_freq= 0, feature_fraction= 1), 
                       param_grid = params_test4, scoring='roc_auc',cv=10,n_jobs=-1)
gsearch4.fit(X_train,y_train)
gsearch4.best_params_, gsearch4.best_score_    # 'lambda_l1': 0.3, 'lambda_l2': 0.5

params_test5={'min_split_gain':[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}
gsearch5 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=52, max_depth=4, num_leaves=16,max_bin=15,min_data_in_leaf=71,
                                                       bagging_fraction=0.6,bagging_freq= 0, feature_fraction= 1,
                        lambda_l1=0.3,lambda_l2=0.5), param_grid = params_test5, scoring='roc_auc',cv=10,n_jobs=-1)
gsearch5.fit(X_train,y_train)
gsearch5.best_params_, gsearch5.best_score_    # 'min_split_gain': 0.0

#%%
# 建立最終模型
model = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.01, n_estimators=1000, max_depth=-1, num_leaves=31,max_bin=75,min_data_in_leaf=31,bagging_fraction=0.6,bagging_freq= 0, feature_fraction= 0.8,
                        lambda_l1=0.001,lambda_l2=0.001, min_split_gain=0)
model.fit(X_train, y_train)
y_pre=model.predict(X_test)
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn import metrics
print("acc:",metrics.accuracy_score(y_test,y_pre))
print("auc:",metrics.roc_auc_score(y_test,y_pre))
data_test6 = pd.read_csv('test6.csv', encoding = 'Big5')
data_test6 = data_test6.replace(['Y', 'N'], [1, 0])
for i in data_test6.columns:
    col_type = data_test6[i].dtype
    if col_type == 'object' or col_type.name == 'category':
        data_test6[i] = data_test6[i].astype('category')
data_test6.info()
X.info()
probabilities = model.predict_proba(data_test6)

type(probabilities)
buy_prob = probabilities[:,1]
will_buy = buy_prob[buy_prob > 0.4]    
# 目前最高'max_depth': -1, 'num_leaves': 31, 97will buy
#%%
# 將機率寫入csv檔中
import os
import pandas as pd
os.chdir('C:\\Users\\方永騰\\Desktop\\研究所\\碩一上\\國泰大數據競賽\\預測機率')
type(buy_prob)
dataframe = pd.DataFrame(buy_prob)
dataframe.to_csv("lgbmclassifiertest01.csv",index=False,sep=',')
