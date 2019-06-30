# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:07:33 2019

@author: kghiy
"""

import pandas as pd

train_set = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")

train_set = train_set.dropna(thresh=800, axis=1)
test_set = test_set.dropna(thresh=800, axis=1)

#import seaborn as sns
#import matplotlib.pyplot as plt

#print(sns.FacetGrid(train_set, col='Street').map(plt.hist, 'SalePrice', bins=20))
    

#Drop useless columns

saleprice = train_set.SalePrice
test_id = test_set['Id']

for col in train_set.columns:
    if(len(train_set[col].unique()) > 9):
        train_set = train_set.drop(col, axis=1)
    elif(train_set[col].value_counts().max() > 1175):
        train_set = train_set.drop(col, axis=1)
        
train_set['SalePrice'] = saleprice

for col in test_set.columns:
    if(len(test_set[col].unique()) > 9):
        test_set = test_set.drop(col, axis=1)      
    elif(test_set[col].value_counts().max() > 1175):
        test_set = test_set.drop(col, axis=1)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = 'most_frequent')
train_set = pd.DataFrame(imputer.fit_transform(train_set), columns=train_set.columns)
test_set = pd.DataFrame(imputer.fit_transform(test_set), columns=test_set.columns)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
ints = ['BsmtFullBath', 'FullBath','HalfBath', 'BedroomAbvGr','Fireplaces',
        'GarageCars', 'YrSold', 'SalePrice']
objects = [x for x in train_set.columns if x not in ints]

#for col in objects:
#    train_set[col] = label_encoder.fit_transform(train_set[col])
#    test_set[col] = label_encoder.fit_transform(test_set[col])
    
train_set = train_set.drop(['SalePrice'], axis=1)
train_set = pd.get_dummies(train_set)
#train_set.insert(78, column='FullBath_4', value=0)
test_set = pd.get_dummies(test_set)
train_set = train_set[train_set.columns]
train_set['SalePrice'] = saleprice

#train_set[ints] = train_set[ints].astype(int)
#ints.remove('SalePrice')
#test_set[ints] = test_set[ints].astype(int)
    
X_train = train_set.iloc[:, train_set.columns != 'SalePrice']
y_train = train_set["SalePrice"]
X_test = test_set

from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=70)
my_model.fit(X_train, y_train, verbose=False)

y_test = my_model.predict(X_test)

X_test['Id'] = test_id 

submission = pd.DataFrame({
        "Id": X_test["Id"],
        "SalePrice": y_test
    })
submission.to_csv('submission.csv', index=False)