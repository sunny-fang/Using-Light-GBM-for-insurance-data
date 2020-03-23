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
os.chdir('C:\\Users\\方永騰\\Desktop\\研究所\\碩一上\\國泰大數據競賽\\刪變數後資料集')

#%%
# LGBMClassifier
data_train6 = pd.read_csv('train6.csv', encoding = 'Big5')
#data_train4 = data_train4.drop(['Unnamed: 0'], axis = 1)
#data_train3 = pd.read_csv('train3.csv', encoding = 'Big5')
train6_y1 = data_train6['Y1']
#train3_y1 = train3_y1.replace(['Y', 'N'], [1, 0])
#train_data = pd.concat([data_train4, train3_y1], axis = 1)
#type(data)
#data = data.replace(['Y', 'N'], [1, 0])
target = train6_y1
X = data_train6.drop(['Y1'], axis = 1)

# Transform categorical features into the appropriate type that is expected by LightGBM
for c in X.columns:
    col_type = X[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        X[c] = X[c].astype('category')
       
# Printout types of features in the dataset
X.info()
X.columns
x_categorical = X.dtypes == 'category'
x_categorical = DataFrame(x_categorical)
aa = x_categorical[x_categorical[0] == True]
type(aa)
categorical_var = list(aa.index)

X_train,X_test,y_train,y_test =train_test_split(X,target,test_size=0.2)
fit_params={"early_stopping_rounds":10, 
            "eval_metric" : 'auc', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            'verbose': 100,
            'feature_name': 'auto', # that's actually the default
            'categorical_feature': 'auto' # that's actually the default
           }
#%%
# 初版(不要動)，auc = 0.818532，>0.4，90個會買
clf = lgb.LGBMClassifier(num_leaves= 15, max_depth=-1, 
                         random_state=314, 
                         silent=True, 
                         metric='None', 
                         n_jobs=4, 
                         n_estimators=1000,
                         colsample_bytree=0.9,
                         subsample=0.9,
                         learning_rate=0.1)
clf.fit(X_train, y_train, **fit_params)
#%%
# 調參數:https://www.cnblogs.com/bjwu/p/9307344.html
clf = lgb.LGBMClassifier(num_leaves= 50, max_depth=6, objective = 'regression', 
                         random_state=314, 
                         silent=True, 
                         metric='rmse', 
                         n_jobs=4, 
                         n_estimators=1000,
                         colsample_bytree=0.8,
                         subsample=0.8,
                         learning_rate=0.1)
clf.fit(X_train, y_train, **fit_params)

# 用LightGBM的cv函數進行演示
params = {
    'boosting_type': 'gbdt', 
    'objective': 'regression', 
    'learning_rate': 0.1, 
    'num_leaves': 50, 
    'max_depth': 6,
    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    }
data_train = lgb.Dataset(X_train, y_train, categorical_feature = categorical_var)
cv_results = lgb.cv(
    params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
    early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)
print('best n_estimators:', len(cv_results['rmse-mean']))    # 以n_est.=26跑
print('best cv score:', cv_results['rmse-mean'][-1])

#%%
# 利用GridSearchCV調參，以LGBMRegressor為例
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,
                              learning_rate=0.1, n_estimators=26, max_depth=6,
                              metric='rmse', bagging_fraction = 0.8,feature_fraction = 0.8)
params_test1={
    'max_depth': range(3,8,2),
    'num_leaves':range(50, 170, 30)
}
gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch1.fit(X_train, y_train)
gsearch1.best_params_, gsearch1.best_score_    # grid_scores_無法跑，原因未知，結果為(5,50)

# 第二次，比較精密的測試
params_test2={
    'max_depth': [4,5,6],
    'num_leaves':[24,30,36,44,50,56,62,68,74]
}
gsearch2 = GridSearchCV(estimator=model_lgb, param_grid=params_test2, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch2.fit(X_train, y_train)
gsearch2.best_params_, gsearch2.best_score_    # 結果為(6,24)

# 其他參數
params_test3={
    'min_child_samples': [16,18,20,22,24],
    'min_child_weight':[0.5,1,1.5,2,2.5,3]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=24,
                              learning_rate=0.1, n_estimators=43, max_depth=6, 
                              metric='rmse', bagging_fraction = 0.8, feature_fraction = 0.8)
gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test3, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch3.fit(X_train, y_train)
gsearch3.best_params_, gsearch3.best_score_    # 結果為(24,0.5)，第二次為(32,0.05)，下面為第三次
params_test3_1={
    'min_child_samples': [32,38,44,50,56],
    'min_child_weight':[0.001,0.005,0.01,0.025,0.05]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=24,
                              learning_rate=0.1, n_estimators=43, max_depth=6, 
                              metric='rmse', bagging_fraction = 0.8, feature_fraction = 0.8)
gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test3_1, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch3.fit(X_train, y_train)
gsearch3.best_params_, gsearch3.best_score_    # 結果為(32,0.001)
params_test4={
    'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=24,
                              learning_rate=0.1, n_estimators=43, max_depth=6, 
                              metric='rmse', bagging_freq = 5,  min_child_samples=21)
gsearch4 = GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch4.fit(X_train, y_train)
gsearch4.best_params_, gsearch4.best_score_    # 結果為'bagging_fraction': 0.9, 'feature_fraction': 0.7

# 重新建模
model = lgb.LGBMRegressor(objective='regression',num_leaves=24,
                          learning_rate=0.005, n_estimators=26, max_depth=6,
                          metric='rmse', bagging_fraction = 0.9,feature_fraction = 0.7, 
                          min_child_weight=0.001,
                          min_child_samples=32)

model.fit(X_train, y_train, **fit_params)
probabilities = model.predict(data_test4)
type(probabilities)
will_buy = probabilities > 0.4

#%%
# 預測真實資料
data_test6 = pd.read_csv('test6.csv', encoding = 'Big5')
#data_test4 = data_test4.drop(['Unnamed: 0'], axis = 1)
for i in data_test6.columns:
    col_type = data_test6[i].dtype
    if col_type == 'object' or col_type.name == 'category':
        data_test6[i] = data_test6[i].astype('category')
data_test6.info()
probabilities = clf.predict_proba(data_test6)

type(probabilities)
buy_prob = probabilities[:,1]
will_buy = buy_prob[buy_prob > 0.4]
# 寫入csv檔中
os.chdir('C:\\Users\\方永騰\\Desktop\\研究所\\碩一上\\國泰大數據競賽\\預測機率')
type(buy_prob)
dataframe = pd.DataFrame(buy_prob)
dataframe.to_csv("lbgmclassifiertest01.csv",index=False,sep=',')

#%%
# LGBMRegressor，和上面不同
# gbm = lgb.LGBMRegressor(objective='regression',num_leaves=31,learning_rate=0.05,n_estimators=20)
# gbm.fit(X_train, y_train,eval_set=[(X_test, y_test)],eval_metric='l1',early_stopping_rounds=5, 
        categorical_feature = 'auto')
gbm.fit()
# 預測訓練集中切出來的測試資料
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# 模型評估
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
# feature importances
print('Feature importances:', list(gbm.feature_importances_))

# 預測真實資料
test_data = pd.read_csv('test4.csv', encoding = 'Big5')
y_pred = gbm.predict(test_data, num_iteration=gbm.best_iteration_)

