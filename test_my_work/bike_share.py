# -*- coding: utf-8 -*-
import sys
sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
from environment.BikeShareEnv import BikeShareEnv
from utils import init_seed
from test_my_work import run_agent


if __name__ == '__main__':
    init_seed.init_seed()
    # print(BikeShareEnv(50).base_score)
    run_agent.run(BikeShareEnv(30), 'PPO_BikeShare_30_x1000', False)
