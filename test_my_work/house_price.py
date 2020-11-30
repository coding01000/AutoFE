# -*- coding: utf-8 -*-
import sys
sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
from environment.HousePriceEnv import HousePriceEnv
from utils import init_seed
from test_my_work import run_agent


if __name__ == '__main__':
    init_seed.init_seed()
    # print(HousePriceEnv(80).base_score)
    run_agent.run(HousePriceEnv(80), 'PPO_HousePrice_80', True)
    # main(TitanicEnv(15), 'PPO_Titanic_15', False)
    # print(BikeShareEnv(15).base_score)
    # print(HousePriceEnv(80).base_score)
    # print( HousePriceEnv(80).base_score)
