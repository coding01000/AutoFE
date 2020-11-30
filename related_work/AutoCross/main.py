import sys

from sklearn.preprocessing import OneHotEncoder

sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
from related_work.AutoCross.discretization.discretization import discretization
from utils.init_seed import init_seed
from related_work.AutoCross.beam_search.beam_search import search
from related_work.AutoCross.feature_wise.model_train import evaluation_with_auc
import scipy
# from dataprocessing.house_price_data import get_data
from dataprocessing import dataset
features = [['num_var4', 'num_var22_ult1'], ['var15', 'saldo_medio_var5_ult1__2'], ['num_var42_0', 'saldo_var5__3'],
            ['num_aport_var13_ult1', 'num_var22_ult1'], ['num_op_var39_ult1', 'var36'],
            ['saldo_var37', 'num_var43_recib_ult1'],
            ['var36', 'num_meses_var5_ult3', 'num_op_var39_ult1'], ['num_var42_0', 'saldo_var5__3', 'ind_var5_0'],
            ['num_meses_var5_ult3', 'num_var45_ult1'],
            ['num_trasp_var11_ult1', 'num_aport_var13_ult1', 'num_var22_ult1'],
            ['num_var30', 'num_var22_ult3'], ['num_meses_var5_ult3', 'num_var45_ult1', 'imp_trans_var37_ult1']]
if __name__ == '__main__':
    init_seed()
    df, label = dataset.get_customer_satisfaction()
    train = None
    granularity = {'1':10, '2':100, '3':200}
    columns = df.columns
    print('---start---')
    for feature in features:
        print(feature)
        if feature[0] in columns:
            tmp = df[feature[0]]
        else:
            t = feature[0].split('__')[0]
            g = feature[0].split('__')[1]
            tmp = discretization(df[[t]], granularity[g])
        for i in range(1, len(feature)):
            if feature[i] in columns:
                tmp += df[feature[i]]
            else:
                t = feature[i].split('__')[0]
                g = feature[i].split('__')[1]
                tmp += discretization(df[[t]], granularity[g])
        ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')
        tmp = ohe.fit_transform(tmp.values.reshape(-1, 1))
        train = scipy.sparse.hstack([train, tmp])
    print(train.shape)
    print(evaluation_with_auc(train, label))


