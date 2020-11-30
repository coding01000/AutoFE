# -*- coding: utf-8 -*-
import sys
sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
from utils import init_seed
from test_my_work import run_agent
import environment as env


if __name__ == '__main__':
    init_seed.init_seed()
    environment = env.MarketEnv(20)
    print(environment.base_score)
    run_agent.run(environment, 'PPO_Market_20', True)
    # main(TitanicEnv(15), 'PPO_Titanic_15', False)
    # print(BikeShareEnv(15).base_score)
    # print(HousePriceEnv(80).base_score)
    # print( HousePriceEnv(80).base_score)
