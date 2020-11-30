# -*- coding: utf-8 -*-
import sys
sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
from test_my_work.no_cross.environment.CustomerEnv import CustomerEnv
from utils import init_seed
from test_my_work import run_agent


if __name__ == '__main__':
    init_seed.init_seed()
    customer = CustomerEnv(80)
    print(customer.base_score)
    run_agent.run(customer, 'NO_CROSS_Customer_80', False)
