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
    # train = train.sample(frac=1.0)
    # train_len = int(len(train) * 0.8)
    # if is_test is False:
    #     train = train[:train_len]
    # else:
    #     train = train[train_len:]
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
    # aaa
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
    # train = train[:1000]
    return train, label


def get_credit_card():
    dir = os.path.join(basepath, 'CreditCard')
    train = pd.read_csv(os.path.join(dir, 'train.csv'))

    label = train['default.payment.next.month']
    train.drop('default.payment.next.month', axis=1, inplace=True)
    train.drop('ID', axis=1, inplace=True)
    return train, label


def get_bank():
    dir = os.path.join(basepath, 'Bank')
    train = pd.read_csv(os.path.join(dir, 'train.csv'))
    for i in train.columns:
        train[i] = LabelEncoder().fit_transform(train[i])
    label = train['y']
    train.drop('y', axis=1, inplace=True)
    b = []
    for i in train.columns:
        b.append(i.replace('.', ''))
    train.columns = b
    return train, label


def get_property_inspection():
    dir = os.path.join(basepath, 'PropertyInspection')
    train = pd.read_csv(os.path.join(dir, 'train.csv'))
    for i in train.columns:
        if type(train[i][1]) is str:
            train[i] = LabelEncoder().fit_transform(train[i])
    label = train['Hazard']
    train.drop('Hazard', axis=1, inplace=True)
    train.drop('Id', axis=1, inplace=True)

    return train, label


def get_store_sales():
    dir = os.path.join(basepath, 'StoreSales')
    train = pd.read_csv(os.path.join(dir, 'train.csv'), parse_dates=True, low_memory=False, index_col='Date')
    store = pd.read_csv(os.path.join(dir, 'store.csv'), low_memory=False)

    train['Year'] = train.index.year
    train['Month'] = train.index.month
    train['Day'] = train.index.day
    train['WeekOfYear'] = train.index.weekofyear

    store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace=True)
    for i in store.columns:
        if store[i].isna().any():
            if store[i].dtype == 'object':
                store[i].fillna('0', inplace=True)
            else:
                store[i].fillna(0, inplace=True)

    train = pd.merge(train, store, how='inner', on='Store')
    label = train['Sales']

    for i in train.columns:
        if type(train[i][1]) is str:
            train[i] = LabelEncoder().fit_transform(train[i])

    del train['Sales']

    return train, label


def get_car():
    dir = os.path.join(basepath, 'Car')
    train = pd.read_csv(os.path.join(dir, 'train.csv'))
    train.fillna(0, inplace=True)
    for i in train.columns:
        if train[i].dtype == 'object':
            train[i] = LabelEncoder().fit_transform(train[i].apply(str))
    label = train['MSRP']
    train.drop('MSRP', axis=1, inplace=True)

    return train, label


def get_market_sales():
    dir = os.path.join(basepath, 'MarketSales')
    train = pd.read_csv(os.path.join(dir, 'train.csv'))
    train.fillna(0, inplace=True)
    for i in train.columns:
        if train[i].dtype == 'object':
            train[i] = LabelEncoder().fit_transform(train[i].apply(str))
    label = train['Item_Outlet_Sales']
    train.drop('Item_Outlet_Sales', axis=1, inplace=True)

    return train, label


def get_marketing_target():
    dir = os.path.join(basepath, 'MarketingTarget')
    train = pd.read_csv(os.path.join(dir, 'train.csv'), sep=';')
    train.fillna(0, inplace=True)
    for i in train.columns:
        if train[i].dtype == 'object':
            train[i] = LabelEncoder().fit_transform(train[i].apply(str))
    label = train['y']
    train.drop('y', axis=1, inplace=True)

    return train, label


def get_house_sales():
    dir = os.path.join(basepath, 'HouseSales')
    train = pd.read_csv(os.path.join(dir, 'train.csv'), parse_dates=True)

    train['date'] = pd.to_datetime(train['date'])
    train['year'] = train['date'].dt.year
    train['month'] = train['date'].dt.month
    train['day'] = train['date'].dt.day

    label = train['price']
    train.drop('price', axis=1, inplace=True)
    train.drop('id', axis=1, inplace=True)
    train.drop('date', axis=1, inplace=True)
    for i in train.columns:
        if train[i].dtype == 'object':
            train[i] = LabelEncoder().fit_transform(train[i].apply(str))

    return train, label


# if __name__ == '__main__':
#     train, label = get_house_sales()
#     train['date'] = pd.to_datetime(train['date'])
#     print(train['date'])
#     print(train.date.dt.year)
#     print(train.isna().any().any())
#     for i in train.columns:
#         print(i, train[i].isna().any(), train[i].dtype == 'object')
    # print(label.isna().any())
#     dir = os.path.join(basepath, 'StoreSales')
#     train = pd.read_csv(os.path.join(dir, 'train.csv'), parse_dates=True, low_memory=False, index_col='Date')
#     store = pd.read_csv(os.path.join(dir, 'store.csv'), low_memory=False)
#
#     train['Year'] = train.index.year
#     train['Month'] = train.index.month
#     train['Day'] = train.index.day
#     train['WeekOfYear'] = train.index.weekofyear
#
#     store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace=True)
#     for i in store.columns:
#         if store[i].isna().any():
#             if store[i].dtype == 'object':
#                 store[i].fillna('0', inplace=True)
#             else:
#                 store[i].fillna(0, inplace=True)
#     print(store.head(10))
    # store.fillna(0, inplace=True)
    # print(store.head(10))
    # print(store.head(10))
#
#     train = pd.merge(train, store, how='inner', on='Store')
    # store = store.replace([1, 2], np.nan)
    # print(store.shape)
    # print(store.dropna(inplace=True))
    # print(store.isnull().any().any())
    # print(train.shape)
    # print(train.isna().any().any())
    # for i in train.columns:
    #     print(f'{i}: {np.isnan(train[i].toarray()).all()}')
    # print('store')
    # for i in store.columns:
    #     print(f'{i}: {np.isnan(store[i].toarray()).all()}')
