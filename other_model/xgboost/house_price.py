import sys
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
from sklearn.model_selection import cross_validate
from other_model.xgboost.model.common import class_model, regress_model
from dataprocessing import dataset


if __name__ == '__main__':
    x, y = dataset.get_house_price(False)
    print(regress_model(x, y))
