# -*- coding: utf-8 -*-
import sys
sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
from test_my_work.only_cross.environment.BankEnv import BankEnv
from utils import init_seed
from test_my_work import run_agent


if __name__ == '__main__':
    init_seed.init_seed()
    print(BankEnv(20).base_score)
    run_agent.run(BankEnv(20), 'ONLY_CROSS_Bank_20', False)
