# -*- coding: utf-8 -*-
import sys
sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
from environment.TitanicEnv import TitanicEnv
from utils import init_seed
from test_my_work import run_agent


if __name__ == '__main__':
    init_seed.init_seed()
    # print(TitanicEnv(15).base_score)
    run_agent.run(TitanicEnv(15), 'PPO_Titanic_15', True)
