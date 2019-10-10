# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:02:33 2019

@author: Root
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy import stats
from scipy.stats import norm, skew

#Read dataset
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

#Save Id
train_Id = data_train["Id"]
test_Id = data_test["Id"] 

#Drop Id columns 
data_train.drop(["Id"], axis = 1, inplace = True)
data_test.drop(["Id"], axis = 1, inplace = True )

data_train.head().T
plt.plot(data_train["GrLivArea"])
plt.boxplot(data_train["MSSubClass"]) 
plt.show() 
data_train["SalePrice"] =np.log1p(data_train["SalePrice"])

#Histograms and Figures
sns.distplot(data_train['SalePrice'])
#skewness and kurtosis
print("Skewness: %f" % data_train['SalePrice'].skew())
print("Kurtosis: %f" % data_train['SalePrice'].kurt())

plt.figure(); sns.distplot(data_train['OverallQual'],kde=False)
plt.figure(); sns.distplot(data_train['SalePrice'],kde=False)
plt.figure(); sns.distplot(data_train['OverallCond'],kde=False)
plt.figure(figsize=(10,10)); stats.probplot(data_train['SalePrice'], plot=plt)

#Correlations
corr = data_train.corr()["SalePrice"]
corr.sort_values(ascending = False)
plt.figure(figsize=(8,8))
sns.boxplot(data_train['OverallQual'], data_train['SalePrice'], color = 'blue')
plt.legend(); plt.xlabel('OverallQual'); plt.ylabel('SalePrice')
plt.figure(); sns.boxplot(data_train['SaleCondition'], data_train['SalePrice'])
plt.figure(); sns.boxplot(data_train['LotShape'], data_train['SalePrice'])
plt.figure(); sns.boxplot(data_train['OverallQual'], data_train['SalePrice'])

fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(data_train['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(data_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# scatter plot grlivarea/saleprice
plt.figure(figsize=[8,6])
plt.scatter(x=data_train['GrLivArea'], y=data_train['SalePrice'])
plt.xlabel('GrLivArea', fontsize=12)
plt.ylabel('SalePrice', fontsize=12)
#Features 
ntrain = data_train.shape[0]
ntest = data_test.shape[0]
features_train = data_train.drop(["SalePrice"], axis = 1)
y_train = data_train["SalePrice"]
data_features = pd.concat((features_train, data_test)).reset_index(drop = True)

#Number of categorical columns
cols2 = data_features.columns
cols = data_features.select_dtypes([np.number]).columns
str_ = set(cols2)-set(cols)


#Missing Data
data_features.isnull().sum()[data_features.isnull().sum() > 0].sort_values(ascending = False)
#data percent
total_missed = data_features.isnull().sum().sort_values(ascending=False)
percent = (data_features.isnull().sum()/data_features.isnull().count()).sort_values(ascending=False)
data_missed = pd.concat([total_missed, percent], axis=1, keys=['Total Missed', 'Percent'])
data_missed.head(10)

#Visualize missing Data
plt.figure(figsize=[20,5])
plt.xticks(rotation='90', fontsize=14)
sns.barplot(x=data_missed.index, y=data_missed.Percent)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
# get unique values of the column data
data_features['PoolQC'].unique()
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual',
            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"PoolQC"
           ,'Alley','Fence','MiscFeature','FireplaceQu','MasVnrType','Utilities']:
    data_features[col] = data_features[col].fillna('None')
    

"""for col in ['GarageYrBlt', 'GarageArea', 'GarageCars','MasVnrArea','BsmtFinSF1','BsmtFinSF2'
           ,'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BsmtUnfSF','TotalBsmtSF']:
    aa = data_features[col][data_features[col] == 0].count()
    print("col : ", "aa" )"""

# Replacing missing data with 0 
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','MasVnrArea','BsmtFinSF1','BsmtFinSF2'
           ,'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BsmtUnfSF','TotalBsmtSF'):
    data_features[col] = data_features[col].fillna(0)
    
data_features["LotFrontage"] = data_features.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

data_features['MSZoning'] = data_features['MSZoning'].fillna(data_features['MSZoning'].mode()[0])
#according to description NA means typical
data_features["Functional"] = data_features["Functional"].fillna("Typ")
#This features has mostly SBrkr so we can fillna for the missing value 
data_features['Electrical'] = data_features['Electrical'].fillna(data_features['Electrical'].mode()[0])
data_features['KitchenQual'] = data_features['KitchenQual'].fillna(data_features['KitchenQual'].mode()[0])
data_features['Exterior1st'] = data_features['Exterior1st'].fillna(data_features['Exterior1st'].mode()[0])
data_features['Exterior2nd'] = data_features['Exterior2nd'].fillna(data_features['Exterior2nd'].mode()[0])
data_features['SaleType'] = data_features['SaleType'].fillna(data_features['SaleType'].mode()[0])

#Check remaning missing values if any
data_features.isnull().sum()[data_features.isnull().sum() > 0].sort_values(ascending = False)

data_features['MSSubClass'].unique()
data_features['YrSold']  = data_features['YrSold'].astype(str)
data_features['OverallCond'] = data_features['OverallCond'].astype(str)
data_features['MSSubClass'] = data_features['MSSubClass'].astype(str)
data_features['MoSold'] = data_features['MoSold'].astype(str)
aa = list(data_features.select_dtypes(include=['object']).columns)

numerical_features = data_features.select_dtypes(exclude = ["object"]).columns
num_feat = data_features[numerical_features]
print("Numerical features : " + str(len(numerical_features)))
skewness = num_feat.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 1]
skewness.sort_values(ascending=False)
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    num_feat[feat] = boxcox1p(num_feat[feat], stats.boxcox_normmax(num_feat[feat] +1 ))
    data_features[feat] = boxcox1p(data_features[feat], stats.boxcox_normmax(data_features[feat] + 1))

#label encoding to some ordering categorical variable 
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for col in cols: 
    le = LabelEncoder()
    data_features[col] = le.fit_transform(data_features[col])
   
print('Shape all_data: {}'.format(data_features.shape))
#aa = list(set(aa)-set(list(cols)))
#tuple(aa)

    
#data_features[skewed_features] = np.log1p(data_features[skewed_features])
#Getting dummy categorical features
data_features = pd.get_dummies(data_features)
print(data_features.shape)

train = data_features[:ntrain]
test = data_features[ntrain:]
X_train = train

#Model
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error , make_scorer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor

kfolds = KFold(n_splits=18, shuffle=True, random_state=42)

# model scoring and validation function
def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, train, y_train,scoring="neg_mean_squared_error",cv=kfolds))
    return (rmse)

# rmsle scoring function
def rmsle(y_train, y_pred):
    return np.sqrt(mean_squared_error(y_train, y_pred))


#Lightgbm 
lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=5000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1)
print("LightGBM score: {:.4f} \n".format(cv_rmse(lightgbm).mean()))
#print("Lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
#np.expm1(y_train)

#Lasso
lasso = Lasso(alpha =0.0005, random_state=1)
print("Lasso score: {:.4f} \n".format(cv_rmse(lasso).mean()))

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5,10,5]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

#Ridge
ridge = RidgeCV(alphas=alphas_alt, cv=kfolds)
print("Ridge score: {:.4f} \n".format(cv_rmse(ridge).mean()))

#ElasticNet
elasticnet = ElasticNetCV(cv=kfolds,alphas=e_alphas)
print("ElasticNet score: {:.4f} \n".format(cv_rmse(elasticnet).mean()))

#Svr
#svr = SVR()
#print(cv_rmse(svr).mean())

#XGBoost
xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
print("XGBoost score: {:.4f} \n".format(cv_rmse(xgboost).mean()))

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet,
                                             xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)

elastic_model = elasticnet.fit(X_train, y_train)
lasso_model = lasso.fit(X_train, y_train)
ridge_model = ridge.fit(X_train, y_train)
lgb_model = lightgbm.fit(X_train, y_train)
xgboost_model = xgboost.fit(X_train, y_train)
stack_gen_model = stack_gen.fit(np.array(X_train), np.array(y_train))

def blend_models_predict(X_train):
    return ((0.1  * elastic_model.predict(X_train)) + \
            (0.16 * lasso_model.predict(X_train)) + \
            (0.11 * ridge_model.predict(X_train)) + \
            (0.2 * lgb_model.predict(X_train)) + \
            (0.16 * xgboost_model.predict(X_train)) + \
            (0.27 * stack_gen_model.predict(np.array(X_train))))
    
print(rmsle(y_train, blend_models_predict(X_train)))

submission = pd.read_csv("sample_submission.csv")
submission.iloc[:,1] = (np.expm1(blend_models_predict(test)))
submission.to_csv("submission.csv", index=False)
