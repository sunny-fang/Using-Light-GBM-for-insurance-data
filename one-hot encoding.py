#%%
# one-hot encoding
import pandas as pd
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
# 讀資料&預處理
os.chdir('D:\\...')
data = pd.read_csv('train6.csv', encoding = 'Big5')
data['Y1'] = data['Y1'].replace(['Y', 'N'], [1, 0])
#%%
# 定義
def get_dummies(dummy, dataset):
    ''''
    make variables dummies
    ref：http://blog.csdn.net/weiwei9363/article/details/78255210
    '''
    dummy_fields = list(dummy)
    for each in dummy_fields:
        dummies = pd.get_dummies( dataset.loc[:, each], prefix=each ) 
        dataset = pd.concat( [dataset, dummies], axis = 1 )
    
    fields_to_drop = dummy_fields
    dataset = dataset.drop( fields_to_drop, axis = 1 )
    return dataset
# 轉換原始資料
new_data = get_dummies(['INSD_1ST_AGE'], data)
for i in new_data.columns:
    col_type = new_data[i].dtype
    if col_type == 'object' or col_type.name == 'category':
        new_data[i] = new_data[i].astype('category')
#%%
X = new_data.drop(['Y1'], axis = 1)
target = new_data['Y1']
X_train,X_test,y_train,y_test =train_test_split(X,target,test_size=0.2, random_state=0)
params = {
    'boosting_type': 'gbdt', 
    'objective': 'binary', 
    'metric': 'auc',
    'learning_rate': 0.1, 
    'num_leaves': 200, 
    'max_depth': 8,
    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    }    
data_train = lgb.Dataset(X_train, y_train, silent=True)
cv_results = lgb.cv(
    params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
    early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)

print('best n_estimators:', len(cv_results['rmse-mean']))# best n_estimators: 13
print('best cv score:', cv_results['rmse-mean'][-1])
#%%
# 開始調參
from sklearn.model_selection import GridSearchCV
params_test1={'max_depth': range(3,8,1), 'num_leaves':range(5, 100, 5)}    
params_test1={'max_depth': [7], 'num_leaves':range(18, 22, 1)} 
gsearch1 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=13), 
                       param_grid = params_test1, scoring='roc_auc',cv=10,n_jobs=-1)
gsearch1.fit(X_train,y_train)
gsearch1.best_params_, gsearch1.best_score_    # 'max_depth': 7, 'num_leaves': 20

params_test2={'max_bin': range(5,256,10), 'min_data_in_leaf':range(1,102,10)}
params_test2={'max_bin': range(133,138,1), 'min_data_in_leaf':range(89,94,1)}
gsearch2 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=13, max_depth=7, num_leaves=20), 
                       param_grid = params_test2, scoring='roc_auc',cv=10,n_jobs=-1)
gsearch2.fit(X_train,y_train)
gsearch2.best_params_, gsearch1.best_score_    # 'max_bin': 135, 'min_data_in_leaf': 91

params_test3={'feature_fraction': [0.6,0.7,0.8,0.9,1.0],
              'bagging_fraction': [0.6,0.7,0.8,0.9,1.0],
              'bagging_freq': range(0,81,10)
}
gsearch3 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=13, max_depth=7, num_leaves=20, max_bin=135, min_data_in_leaf=91), 
                       param_grid = params_test3, scoring='roc_auc',cv=10,n_jobs=-1)
gsearch3.fit(X_train,y_train)
gsearch3.best_params_, gsearch3.best_score_    # 'bagging_fraction': 0.6, 'bagging_freq': 0, 'feature_fraction': 1

params_test4={'lambda_l1': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0],
              'lambda_l2': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]
}     
gsearch4 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=13, max_depth=7, num_leaves=20,max_bin=135, min_data_in_leaf=91, 
                                                       bagging_fraction=0.6,bagging_freq= 0, feature_fraction= 1), 
                       param_grid = params_test4, scoring='roc_auc',cv=10,n_jobs=-1)
gsearch4.fit(X_train,y_train)
gsearch4.best_params_, gsearch4.best_score_    # 'lambda_l1': 0.0, 'lambda_l2': 0.0

params_test5={'min_split_gain':[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}
gsearch5 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=13, max_depth=7, num_leaves=20,max_bin=135, min_data_in_leaf=91,
                                                       bagging_fraction=0.6,bagging_freq= 0, feature_fraction= 1,
                        lambda_l1=0.0,lambda_l2=0.0), param_grid = params_test5, scoring='roc_auc',cv=10,n_jobs=-1)
gsearch5.fit(X_train,y_train)
gsearch5.best_params_, gsearch5.best_score_    # 'min_split_gain': 0.0
#%%
# 再測一次n_eatimator
params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'nthread':4,
          'learning_rate':0.005,
          'num_leaves':20, 
          'max_depth': 7,   
          'subsample': 0.8, 
          'colsample_bytree': 0.8, 
          'max_bin':135,
          'min_data_in_leaf':91,
          'bagging_fraction':0.6,'bagging_freq': 0, 'feature_fraction': 1,
          'lambda_l1':0.0, 'lambda_l2':0.0
    }    
data_train = lgb.Dataset(X_train, y_train, silent=True)
cv_results = lgb.cv(
    params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
    early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)

print('best n_estimators:', len(cv_results['rmse-mean']))# best n_estimators: 原為13，變937
print('best cv score:', cv_results['rmse-mean'][-1])
#%%
data_test6 = pd.read_csv('test6.csv', encoding = 'Big5')
data_test6 = get_dummies(['INSD_1ST_AGE'], data_test6)# 把data_test6 one-hot encoding
for i in data_test6.columns:
    col_type = data_test6[i].dtype
    if col_type == 'object' or col_type.name == 'category':
        data_test6[i] = data_test6[i].astype('category')
#%%
model = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.005, 
                           n_estimators=937, max_depth=7, num_leaves=20,max_bin=135,min_data_in_leaf=91,bagging_fraction=0.6,bagging_freq= 0, feature_fraction= 1,
                           lambda_l1=0.0,lambda_l2=0.0, min_split_gain=0)
model.fit(X_train, y_train)
probabilities = model.predict_proba(data_test6)
buy_prob = probabilities[:,1]
will_buy = buy_prob[buy_prob > 0.4]  
#%%
os.chdir('D:\\...')
type(buy_prob)
dataframe = pd.DataFrame(buy_prob)
dataframe.to_csv("lgbmclassifiertest04.csv",index=False,sep=',')
