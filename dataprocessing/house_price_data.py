import pandas as pd
import numpy as np


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


# def get_data(is_test):
#     train = get_data_(pd.read_csv('dataset/HousePrice/train.csv'))
#     if is_test:
#         test = get_data_(pd.read_csv('dataset/HousePrice/test.csv'))
#         return train, test
#
#     label = train['SalePrice'].apply(np.log1p)
#     train.drop('Id', axis=1, inplace=True)
#     train.drop('SalePrice', axis=1, inplace=True)
#     return train, label
