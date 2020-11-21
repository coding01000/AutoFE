# from dataprocessing.house_price_data import get_data, get_data_
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os
import numpy as np
basepath = os.path.abspath(__file__)
basepath = os.path.dirname(os.path.dirname(basepath))
basepath = os.path.join(basepath, 'dataset')


def get_house_price(is_test):
    dir = os.path.join(basepath, 'HousePrice')
    train = get_data_(pd.read_csv(os.path.join(dir, 'train.csv')))
    if is_test:
        test = get_data_(pd.read_csv(os.path.join(dir, 'test.csv')))
        return train, test

    label = train['SalePrice'].apply(np.log1p)
    train.drop('Id', axis=1, inplace=True)
    train.drop('SalePrice', axis=1, inplace=True)
    return train, label


def get_amazon(is_test):
    dir = os.path.join(basepath, 'Amazon')
    train = pd.read_csv(os.path.join(dir, 'train.csv'))
    test = pd.read_csv(os.path.join(dir, 'test.csv'))
    if is_test:
        return train, test

    label = train['ACTION']
    train.drop('ACTION', axis=1, inplace=True)
    return train, label


def get_bike_share(is_test):
    dir = os.path.join(basepath, 'BikeShare')
    train = pd.read_csv(os.path.join(dir, 'train.csv'))
    target = ['registered', 'casual']
    col4train = [x for x in train.columns if x not in target]
    train = train[col4train]
    train['datetime'] = pd.to_datetime(train['datetime']).dt.hour
    if is_test:
        test = pd.read_csv(os.path.join(dir, 'test.csv'))
        test = test[col4train]
        test['datetime'] = pd.to_datetime(test['datetime']).dt.hour
        return train, test

    label = train['count']
    train.drop('count', axis=1, inplace=True)
    return train, label


def get_titanic(is_test):
    dir = os.path.join(basepath, 'Titanic')
    train = pd.read_csv(os.path.join(dir, 'train.csv'))
    target = ['Ticket', 'Name', 'Cabin']
    col4train = [x for x in train.columns if x not in target]
    train = train[col4train]
    em = LabelEncoder()
    sex = LabelEncoder()
    train['Embarked'] = em.fit_transform(train['Embarked'].astype(str))
    train['Sex'] = sex.fit_transform(train['Sex'])
    train = train.fillna(0)
    if is_test:
        test = pd.read_csv(os.path.join(dir, 'test.csv'))
        test = test[col4train]
        test['Embarked'] = em.transform(test['Embarked'].astype(str))
        test['Sex'] = sex.transform(test['Sex'])
        test = test.fillna(0)
        return train, test

    label = train["Survived"]
    train.drop('Survived', axis=1, inplace=True)
    return train, label


def get_data_(train):
    # train = pd.read_csv('dataset/HousePrice/train.csv')
    train["PoolQC"] = train["PoolQC"].fillna("None")
    train["MiscFeature"] = train["MiscFeature"].fillna("None")
    train["Alley"] = train["Alley"].fillna("None")
    train["Fence"] = train["Fence"].fillna("None")
    train["FireplaceQu"] = train["FireplaceQu"].fillna("None")
    train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        train[col] = train[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        train[col] = train[col].fillna(0)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        train[col] = train[col].fillna(0)
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        train[col] = train[col].fillna('None')
    train["MasVnrType"] = train["MasVnrType"].fillna("None")
    train["MasVnrArea"] = train["MasVnrArea"].fillna(0)
    train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])
    train = train.drop(['Utilities'], axis=1)
    train["Functional"] = train["Functional"].fillna("Typ")
    train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
    train['KitchenQual'] = train['KitchenQual'].fillna(train['KitchenQual'].mode()[0])
    train['Exterior1st'] = train['Exterior1st'].fillna(train['Exterior1st'].mode()[0])
    train['Exterior2nd'] = train['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0])
    train['SaleType'] = train['SaleType'].fillna(train['SaleType'].mode()[0])
    train['MSSubClass'] = train['MSSubClass'].fillna("None")

    from sklearn.preprocessing import LabelEncoder
    cols = train.select_dtypes(include='object').columns

    for c in cols:
        lbl = LabelEncoder()
        train[c] = lbl.fit_transform(train[c].astype(str))
    # label = train['SalePrice']
    # train.drop('Id', axis=1, inplace=True)
    # train.drop('SalePrice', axis=1, inplace=True)
    return train


def get_customer_satisfaction():
    dir = os.path.join(basepath, 'CustomerSatisfaction')
    train = pd.read_csv(os.path.join(dir, 'train.csv'))
    remove = []
    for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)
    train.drop(remove, axis=1, inplace=True)

    remove = []
    c = train.columns
    for i in range(len(c) - 1):
        v = train[c[i]].values
        for j in range(i + 1, len(c)):
            if np.array_equal(v, train[c[j]].values):
                remove.append(c[j])
    train.drop(remove, axis=1, inplace=True)

    label = train['TARGET']
    train.drop('TARGET', axis=1, inplace=True)
    train.drop('ID', axis=1, inplace=True)
    return train, label






