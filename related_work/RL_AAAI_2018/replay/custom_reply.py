import sys
from scipy import sparse
import os
sys.path.append(os.path.abspath(__file__).split('AutoFE')[0]+'AutoFE')
from related_work.Explorekit.evaluation import evalution
from utils import init_seed
from test_my_work import run_agent
from multiprocessing import Pool
from dataprocessing import dataset
import environment as env
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
init_seed.init_seed()
base_score = env.CreditEnv(20).base_score
print(base_score)
run_agent.run(env.CreditEnv(20), 'PPO_CreditEnv_20', False)
# np.set_printoptions(threshold=np.inf)
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# print(dataset.get_amazon(False)[0].shape)
# print(dataset.get_house_price(False)[0].shape)
# print(dataset.get_titanic(False)[0].shape)
# print(dataset.get_bike_share(False)[0].shape)
# print(dataset.get_customer_satisfaction()[0].shape)
# env = CustomEnv(5)
# # print('---start---')
# print(env.base_score)
# print(env.step(1))
# print(env.step(1))
# # # print(env.step(6))
# # # print(env.step(11))
# # print(env.action.data.columns.values)
# df, label = dataset.get_customer_satisfaction()
# print(env.action.data_sets().data)
# print(evalution.auc_score(env.action.data_sets(), label))
df, label = dataset.get_customer_satisfaction()
features = ['var3','var15','imp_ent_var16_ult1','imp_op_var39_comer_ult1'
,'imp_op_var39_comer_ult3','imp_op_var40_comer_ult1'
,'imp_op_var40_comer_ult3','imp_op_var40_efect_ult3','imp_op_var40_ult1'
,'imp_op_var41_comer_ult1','imp_op_var41_comer_ult3'
,'imp_op_var41_efect_ult1','imp_op_var41_efect_ult3','imp_op_var41_ult1'
,'imp_op_var39_efect_ult3','imp_op_var39_ult1','imp_sal_var16_ult1'
,'ind_var1_0','ind_var1','ind_var5_0','ind_var5','ind_var6_0','ind_var8_0'
,'ind_var8','ind_var12_0','ind_var12','ind_var13_0','ind_var13_corto_0'
,'ind_var13_corto','ind_var13_largo_0','ind_var13','ind_var14_0'
,'ind_var14','ind_var17_0','ind_var19','ind_var20_0','ind_var20'
,'ind_var24_0','ind_var24','ind_var25_cte','ind_var26_0','ind_var26_cte'
,'ind_var25_0','ind_var30_0','ind_var30','ind_var31_0','ind_var32_cte'
,'ind_var37_cte','ind_var37_0','ind_var39_0','ind_var41_0','ind_var44_0'
,'num_var1_0','num_var1','num_var4','num_var5_0','num_var5','num_var6_0'
,'num_var8_0','num_var8','num_var12_0','num_var12','num_var13_0'
,'num_var13_corto_0','num_var13_corto','num_var13_largo_0'
,'num_var13_largo','num_var13','num_var14_0','num_var14','num_var17_0'
,'num_var20_0','num_var20','num_var24_0','num_var24','num_var26_0'
,'num_var25_0','num_op_var40_hace2','num_op_var40_hace3'
,'num_op_var40_ult1','num_op_var40_ult3','num_op_var41_hace2'
,'num_op_var41_hace3','num_op_var41_ult1','num_op_var41_ult3'
,'num_op_var39_hace2','num_op_var39_hace3','num_op_var39_ult1'
,'num_op_var39_ult3','num_var30_0','num_var30','num_var31_0','num_var32_0'
,'num_var35','num_var37_med_ult2','num_var37_0','num_var39_0'
,'num_var41_0','num_var42_0','num_var42','num_var44_0','saldo_var1'
,'saldo_var5','saldo_var6','saldo_var8','saldo_var12','saldo_var13_corto'
,'saldo_var13_largo','saldo_var13','saldo_var14','saldo_var17'
,'saldo_var20','saldo_var24','saldo_var26','saldo_var25','saldo_var30'
,'saldo_var31','saldo_var32','saldo_var37','saldo_var42','saldo_var44'
,'var36','delta_imp_aport_var13_1y3','delta_imp_aport_var17_1y3'
,'delta_imp_compra_var44_1y3','delta_imp_reemb_var13_1y3'
,'delta_imp_venta_var44_1y3','imp_aport_var13_hace3'
,'imp_aport_var13_ult1','imp_aport_var17_ult1','imp_var7_recib_ult1'
,'imp_compra_var44_hace3','imp_compra_var44_ult1','imp_reemb_var13_ult1'
,'imp_var43_emit_ult1','imp_trans_var37_ult1','imp_venta_var44_ult1'
,'ind_var7_recib_ult1','ind_var10_ult1','ind_var10cte_ult1'
,'ind_var9_cte_ult1','ind_var9_ult1','ind_var43_emit_ult1'
,'ind_var43_recib_ult1','var21','num_aport_var13_hace3'
,'num_aport_var13_ult1','num_aport_var17_ult1','num_var7_recib_ult1'
,'num_compra_var44_hace3','num_compra_var44_ult1','num_ent_var16_ult1'
,'num_var22_hace2','num_var22_hace3','num_var22_ult1','num_var22_ult3'
,'num_med_var22_ult3','num_med_var45_ult3','num_meses_var5_ult3'
,'num_meses_var8_ult3','num_meses_var12_ult3','num_meses_var13_corto_ult3'
,'num_meses_var13_largo_ult3','num_meses_var17_ult3'
,'num_meses_var39_vig_ult3','num_meses_var44_ult3'
,'num_op_var39_comer_ult1','num_op_var39_comer_ult3'
,'num_op_var40_comer_ult1','num_op_var40_comer_ult3'
,'num_op_var40_efect_ult3','num_op_var41_comer_ult1'
,'num_op_var41_comer_ult3','num_op_var41_efect_ult1'
,'num_op_var41_efect_ult3','num_op_var39_efect_ult3'
,'num_reemb_var13_ult1','num_sal_var16_ult1','num_var43_emit_ult1'
,'num_var43_recib_ult1','num_trasp_var11_ult1','num_var45_hace2'
,'num_var45_hace3','num_var45_ult1','num_var45_ult3'
,'saldo_medio_var5_hace2','saldo_medio_var5_hace3','saldo_medio_var5_ult1'
,'saldo_medio_var5_ult3','saldo_medio_var8_hace2','saldo_medio_var8_hace3'
,'saldo_medio_var8_ult1','saldo_medio_var8_ult3','saldo_medio_var12_hace2'
,'saldo_medio_var12_hace3','saldo_medio_var12_ult1'
,'saldo_medio_var12_ult3','saldo_medio_var13_corto_hace2'
,'saldo_medio_var13_corto_hace3','saldo_medio_var13_corto_ult1'
,'saldo_medio_var13_corto_ult3','saldo_medio_var13_largo_hace2'
,'saldo_medio_var13_largo_hace3','saldo_medio_var13_largo_ult1'
,'saldo_medio_var13_largo_ult3','saldo_medio_var17_hace2'
,'saldo_medio_var17_ult1','saldo_medio_var17_ult3'
,'saldo_medio_var29_ult1','saldo_medio_var44_hace2'
,'saldo_medio_var44_hace3','saldo_medio_var44_ult1'
,'saldo_medio_var44_ult3','var38','var15___1','imp_ent_var16_ult1___1'
,'imp_op_var39_comer_ult1___1','imp_op_var39_comer_ult3___1'
,'imp_op_var40_comer_ult1___1','imp_op_var40_comer_ult3___1'
,'imp_op_var40_efect_ult3___1','imp_op_var40_ult1___1'
,'imp_op_var41_comer_ult1___1','imp_op_var41_comer_ult3___1'
,'imp_op_var41_efect_ult1___1','imp_op_var41_efect_ult3___1'
,'imp_op_var41_ult1___1','imp_op_var39_efect_ult3___1'
,'imp_op_var39_ult1___1','imp_sal_var16_ult1___1','ind_var1_0___1'
,'ind_var1___1','ind_var5_0___1','ind_var5___1','ind_var6_0___1'
,'ind_var8_0___1','ind_var8___1','ind_var12_0___1','ind_var12___1'
,'ind_var13_0___1','ind_var13_corto_0___1','ind_var13_corto___1'
,'ind_var13_largo_0___1','ind_var13___1','ind_var14_0___1','ind_var14___1'
,'ind_var17_0___1','ind_var19___1','ind_var20_0___1','ind_var20___1'
,'ind_var24_0___1','ind_var24___1','ind_var25_cte___1','ind_var26_0___1'
,'ind_var26_cte___1','ind_var25_0___1','ind_var30_0___1','ind_var30___1'
,'ind_var31_0___1','ind_var32_cte___1','ind_var37_cte___1'
,'ind_var37_0___1','ind_var39_0___1','ind_var41_0___1','ind_var44_0___1'
,'num_var1_0___1','num_var1___1','num_var4___1','num_var5_0___1'
,'num_var5___1','num_var6_0___1','num_var8_0___1','num_var8___1'
,'num_var12_0___1','num_var12___1','num_var13_0___1'
,'num_var13_corto_0___1','num_var13_corto___1','num_var13_largo_0___1'
,'num_var13_largo___1','num_var13___1','num_var14_0___1','num_var14___1'
,'num_var17_0___1','num_var20_0___1','num_var20___1','num_var24_0___1'
,'num_var24___1','num_var26_0___1','num_var25_0___1'
,'num_op_var40_hace2___1','num_op_var40_hace3___1','num_op_var40_ult1___1'
,'num_op_var40_ult3___1','num_op_var41_hace2___1','num_op_var41_hace3___1'
,'num_op_var41_ult1___1','num_op_var41_ult3___1','num_op_var39_hace2___1'
,'num_op_var39_hace3___1','num_op_var39_ult1___1','num_op_var39_ult3___1'
,'num_var30_0___1','num_var30___1','num_var31_0___1','num_var32_0___1'
,'num_var35___1','num_var37_med_ult2___1','num_var37_0___1'
,'num_var39_0___1','num_var41_0___1','num_var42_0___1','num_var42___1'
,'num_var44_0___1','saldo_var1___1','saldo_var5___1','saldo_var6___1'
,'saldo_var8___1','saldo_var12___1','saldo_var13_corto___1'
,'saldo_var13_largo___1','saldo_var13___1','saldo_var14___1'
,'saldo_var17___1','saldo_var20___1','saldo_var24___1','saldo_var26___1'
,'saldo_var25___1','saldo_var30___1','saldo_var31___1','saldo_var32___1'
,'saldo_var37___1','saldo_var42___1','saldo_var44___1','var36___1'
,'delta_imp_aport_var13_1y3___1','delta_imp_aport_var17_1y3___1'
,'delta_imp_compra_var44_1y3___1','delta_imp_reemb_var13_1y3___1'
,'delta_imp_venta_var44_1y3___1','imp_aport_var13_hace3___1'
,'imp_aport_var13_ult1___1','imp_aport_var17_ult1___1'
,'imp_var7_recib_ult1___1','imp_compra_var44_hace3___1'
,'imp_compra_var44_ult1___1','imp_reemb_var13_ult1___1'
,'imp_var43_emit_ult1___1','imp_trans_var37_ult1___1'
,'imp_venta_var44_ult1___1','ind_var7_recib_ult1___1','ind_var10_ult1___1'
,'ind_var10cte_ult1___1','ind_var9_cte_ult1___1','ind_var9_ult1___1'
,'ind_var43_emit_ult1___1','ind_var43_recib_ult1___1','var21___1'
,'num_aport_var13_hace3___1','num_aport_var13_ult1___1'
,'num_aport_var17_ult1___1','num_var7_recib_ult1___1'
,'num_compra_var44_hace3___1','num_compra_var44_ult1___1'
,'num_ent_var16_ult1___1','num_var22_hace2___1','num_var22_hace3___1'
,'num_var22_ult1___1','num_var22_ult3___1','num_med_var22_ult3___1'
,'num_med_var45_ult3___1','num_meses_var5_ult3___1'
,'num_meses_var8_ult3___1','num_meses_var12_ult3___1'
,'num_meses_var13_corto_ult3___1','num_meses_var13_largo_ult3___1'
,'num_meses_var17_ult3___1','num_meses_var39_vig_ult3___1'
,'num_meses_var44_ult3___1','num_op_var39_comer_ult1___1'
,'num_op_var39_comer_ult3___1','num_op_var40_comer_ult1___1'
,'num_op_var40_comer_ult3___1','num_op_var40_efect_ult3___1'
,'num_op_var41_comer_ult1___1','num_op_var41_comer_ult3___1'
,'num_op_var41_efect_ult1___1','num_op_var41_efect_ult3___1'
,'num_op_var39_efect_ult3___1','num_reemb_var13_ult1___1'
,'num_sal_var16_ult1___1','num_var43_emit_ult1___1'
,'num_var43_recib_ult1___1','num_trasp_var11_ult1___1'
,'num_var45_hace2___1','num_var45_hace3___1','num_var45_ult1___1'
,'num_var45_ult3___1','saldo_medio_var5_hace2___1'
,'saldo_medio_var5_hace3___1','saldo_medio_var5_ult1___1'
,'saldo_medio_var5_ult3___1','saldo_medio_var8_hace2___1'
,'saldo_medio_var8_hace3___1','saldo_medio_var8_ult1___1'
,'saldo_medio_var8_ult3___1','saldo_medio_var12_hace2___1'
,'saldo_medio_var12_hace3___1','saldo_medio_var12_ult1___1'
,'saldo_medio_var12_ult3___1','saldo_medio_var13_corto_hace2___1'
,'saldo_medio_var13_corto_hace3___1','saldo_medio_var13_corto_ult1___1'
,'saldo_medio_var13_corto_ult3___1','saldo_medio_var13_largo_hace2___1'
,'saldo_medio_var13_largo_hace3___1','saldo_medio_var13_largo_ult1___1'
,'saldo_medio_var13_largo_ult3___1','saldo_medio_var17_hace2___1'
,'saldo_medio_var17_ult1___1','saldo_medio_var17_ult3___1'
,'saldo_medio_var29_ult1___1','saldo_medio_var44_hace2___1'
,'saldo_medio_var44_hace3___1','saldo_medio_var44_ult1___1'
,'saldo_medio_var44_ult3___1','imp_ent_var16_ult1___1___1'
,'imp_op_var40_efect_ult3___1___1','imp_sal_var16_ult1___1___1'
,'ind_var1_0___1___1','ind_var1___1___1','ind_var5_0___1___1'
,'ind_var5___1___1','ind_var6_0___1___1','ind_var8_0___1___1'
,'ind_var8___1___1','ind_var12_0___1___1','ind_var12___1___1'
,'ind_var13_0___1___1','ind_var13_corto_0___1___1'
,'ind_var13_corto___1___1','ind_var13_largo_0___1___1','ind_var13___1___1'
,'ind_var14_0___1___1','ind_var14___1___1','ind_var17_0___1___1'
,'ind_var19___1___1','ind_var20_0___1___1','ind_var20___1___1'
,'ind_var24_0___1___1','ind_var24___1___1','ind_var25_cte___1___1'
,'ind_var26_0___1___1','ind_var26_cte___1___1','ind_var25_0___1___1'
,'ind_var30_0___1___1','ind_var30___1___1','ind_var31_0___1___1'
,'ind_var32_cte___1___1','ind_var37_cte___1___1','ind_var37_0___1___1'
,'ind_var39_0___1___1','ind_var41_0___1___1','ind_var44_0___1___1'
,'num_var1_0___1___1','num_var1___1___1','num_var4___1___1'
,'num_var5_0___1___1','num_var5___1___1','num_var6_0___1___1'
,'num_var8_0___1___1','num_var8___1___1','num_var12_0___1___1'
,'num_var12___1___1','num_var13_0___1___1','num_var13_corto_0___1___1'
,'num_var13_corto___1___1','num_var13_largo_0___1___1'
,'num_var13_largo___1___1','num_var13___1___1','num_var14_0___1___1'
,'num_var14___1___1','num_var17_0___1___1','num_var20_0___1___1'
,'num_var20___1___1','num_var24_0___1___1','num_var24___1___1'
,'num_var26_0___1___1','num_var25_0___1___1','num_op_var40_hace2___1___1'
,'num_op_var40_hace3___1___1','num_op_var40_ult1___1___1'
,'num_op_var40_ult3___1___1','num_op_var41_hace2___1___1'
,'num_op_var41_hace3___1___1','num_op_var41_ult1___1___1'
,'num_op_var41_ult3___1___1','num_op_var39_hace2___1___1'
,'num_op_var39_hace3___1___1','num_op_var39_ult1___1___1'
,'num_op_var39_ult3___1___1','num_var30_0___1___1','num_var30___1___1'
,'num_var31_0___1___1','num_var32_0___1___1','num_var35___1___1'
,'num_var37_med_ult2___1___1','num_var37_0___1___1','num_var39_0___1___1'
,'num_var41_0___1___1','num_var42_0___1___1','num_var42___1___1'
,'num_var44_0___1___1','saldo_var1___1___1','saldo_var5___1___1'
,'saldo_var6___1___1','saldo_var8___1___1','saldo_var12___1___1'
,'saldo_var13_corto___1___1','saldo_var13_largo___1___1'
,'saldo_var13___1___1','saldo_var14___1___1','saldo_var17___1___1'
,'saldo_var20___1___1','saldo_var24___1___1','saldo_var26___1___1'
,'saldo_var25___1___1','saldo_var30___1___1','saldo_var31___1___1'
,'saldo_var32___1___1','saldo_var37___1___1','saldo_var42___1___1'
,'saldo_var44___1___1','var36___1___1','delta_imp_aport_var13_1y3___1___1'
,'delta_imp_aport_var17_1y3___1___1','delta_imp_compra_var44_1y3___1___1'
,'delta_imp_reemb_var13_1y3___1___1','delta_imp_venta_var44_1y3___1___1'
,'imp_aport_var13_hace3___1___1','imp_aport_var13_ult1___1___1'
,'imp_aport_var17_ult1___1___1','imp_var7_recib_ult1___1___1'
,'imp_compra_var44_hace3___1___1','imp_compra_var44_ult1___1___1'
,'imp_reemb_var13_ult1___1___1','imp_var43_emit_ult1___1___1'
,'imp_trans_var37_ult1___1___1','imp_venta_var44_ult1___1___1'
,'ind_var7_recib_ult1___1___1','ind_var10_ult1___1___1'
,'ind_var10cte_ult1___1___1','ind_var9_cte_ult1___1___1'
,'ind_var9_ult1___1___1','ind_var43_emit_ult1___1___1'
,'ind_var43_recib_ult1___1___1','var21___1___1'
,'num_aport_var13_hace3___1___1','num_aport_var13_ult1___1___1'
,'num_aport_var17_ult1___1___1','num_var7_recib_ult1___1___1'
,'num_compra_var44_hace3___1___1','num_compra_var44_ult1___1___1'
,'num_ent_var16_ult1___1___1','num_var22_hace2___1___1'
,'num_var22_hace3___1___1','num_var22_ult1___1___1'
,'num_var22_ult3___1___1','num_med_var22_ult3___1___1'
,'num_med_var45_ult3___1___1','num_meses_var5_ult3___1___1'
,'num_meses_var8_ult3___1___1','num_meses_var12_ult3___1___1'
,'num_meses_var13_corto_ult3___1___1','num_meses_var13_largo_ult3___1___1'
,'num_meses_var17_ult3___1___1','num_meses_var39_vig_ult3___1___1'
,'num_meses_var44_ult3___1___1','num_op_var39_comer_ult1___1___1'
,'num_op_var39_comer_ult3___1___1','num_op_var40_comer_ult1___1___1'
,'num_op_var40_comer_ult3___1___1','num_op_var40_efect_ult3___1___1'
,'num_op_var41_comer_ult1___1___1','num_op_var41_comer_ult3___1___1'
,'num_op_var41_efect_ult1___1___1','num_op_var41_efect_ult3___1___1'
,'num_op_var39_efect_ult3___1___1','num_reemb_var13_ult1___1___1'
,'num_sal_var16_ult1___1___1','num_var43_emit_ult1___1___1'
,'num_var43_recib_ult1___1___1','num_trasp_var11_ult1___1___1'
,'num_var45_hace2___1___1','num_var45_hace3___1___1'
,'num_var45_ult1___1___1','num_var45_ult3___1___1'
,'saldo_medio_var8_hace2___1___1','saldo_medio_var8_hace3___1___1'
,'saldo_medio_var8_ult1___1___1','saldo_medio_var8_ult3___1___1'
,'saldo_medio_var12_hace2___1___1','saldo_medio_var12_hace3___1___1'
,'saldo_medio_var12_ult1___1___1','saldo_medio_var12_ult3___1___1'
,'saldo_medio_var13_corto_hace2___1___1'
,'saldo_medio_var13_corto_hace3___1___1'
,'saldo_medio_var13_corto_ult1___1___1'
,'saldo_medio_var13_corto_ult3___1___1'
,'saldo_medio_var13_largo_hace2___1___1'
,'saldo_medio_var13_largo_hace3___1___1'
,'saldo_medio_var13_largo_ult1___1___1'
,'saldo_medio_var13_largo_ult3___1___1','saldo_medio_var17_hace2___1___1'
,'saldo_medio_var17_ult1___1___1','saldo_medio_var17_ult3___1___1'
,'saldo_medio_var29_ult1___1___1','saldo_medio_var44_hace2___1___1'
,'saldo_medio_var44_hace3___1___1','saldo_medio_var44_ult1___1___1'
,'saldo_medio_var44_ult3___1___1']
columns = df.columns
train = None
print('---start---')
for feature in features:
    print(feature)
    f = feature.split('___')
    tmp = df[[f[0]]]
    for i in range(1, len(f)):
        tmp = tmp.apply(np.square)
    train = sparse.hstack([train, sparse.csr_matrix(tmp)])
print('---train---')
print(train.data)
print(evalution.auc_score(train, label))
