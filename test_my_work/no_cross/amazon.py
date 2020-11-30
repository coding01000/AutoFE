# -*- coding: utf-8 -*-
import sys
sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
from test_my_work.no_cross.environment.AmazonEnv import AmazonEnv
from utils import init_seed
from test_my_work import run_agent


if __name__ == '__main__':
    init_seed.init_seed()
    print(AmazonEnv(15).base_score)
    run_agent.run(AmazonEnv(15), 'NO_CROSS_Amazon_15', False)
