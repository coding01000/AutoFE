# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
from environment.BikeShareEnv import BikeShareEnv
from utils import init_seed
from test_my_work import run_agent
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    init_seed.init_seed()
    print(BikeShareEnv(15).base_score)
    run_agent.run(BikeShareEnv(15), 'PPO_BikeShare_15_x1000', False)
