import sys
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
from other_model.xgboost.model.common import class_model
from dataprocessing import dataset


if __name__ == '__main__':
    x, y = dataset.get_titanic(False)
    print(class_model(x.values, y))
