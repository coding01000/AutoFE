from dataprocessing.house_price_data import get_data
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def get_house_price(is_test):
    return get_data(is_test)


def get_amazon(is_test):
    train = pd.read_csv('dataset/Amazon/train.csv')
    test = pd.read_csv('dataset/Amazon/test.csv')
    if is_test:
        return train, test

    label = train['ACTION']
    train.drop('ACTION', axis=1, inplace=True)
    return train, label


def get_bike_share(is_test):
    train = pd.read_csv('dataset/Bikeshare/train.csv')
    target = ['registered', 'casual']
    col4train = [x for x in train.columns if x not in target]
    train = train[col4train]
    train['datetime'] = pd.to_datetime(train['datetime']).dt.hour
    if is_test:
        test = pd.read_csv('dataset/Bikeshare/test.csv')
        test = test[col4train]
        test['datetime'] = pd.to_datetime(test['datetime']).dt.hour
        return train, test

    label = train['count'].values
    train.drop('count', axis=1, inplace=True)
    return train, label


def get_titanic(is_test):
    train = pd.read_csv('dataset/Titanic/train.csv')
    target = ['Ticket', 'Name', 'Cabin']
    col4train = [x for x in train.columns if x not in target]
    train = train[col4train]
    em = LabelEncoder()
    sex = LabelEncoder()
    train['Embarked'] = em.fit_transform(train['Embarked'].astype(str))
    train['Sex'] = sex.fit_transform(train['Sex'])
    train = train.fillna(0)
    if is_test:
        test = pd.read_csv('dataset/Titanic/test.csv')
        test = test[col4train]
        test['Embarked'] = em.transform(test['Embarked'].astype(str))
        test['Sex'] = sex.transform(test['Sex'])
        test = test.fillna(0)
        return train, test

    label = train["Survived"].values
    train.drop('Survived', axis=1, inplace=True)
    return train, label









