# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath('.'))
import environment as env
from utils import init_seed
from test_my_work import run_agent
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    init_seed.init_seed()
    # print(env.CustomerSatisfactionEnv(15).base_score)
    run_agent.run(env.CustomerSatisfactionEnv(80), 'PPO_CustomerSatisfaction_80', False)
