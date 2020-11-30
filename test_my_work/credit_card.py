import sys
import os
# sys.path.append(os.path.abspath(__file__).split('AutoFE')[0]+'AutoFE')
from test_my_work import run_agent
import environment as env
from utils import init_seed
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# warnings.simplefilter('error', category=UserWarning)

if __name__ == '__main__':
    init_seed.init_seed()
    base_score = env.CreditEnv(20).base_score
    print(base_score)
    run_agent.run(env.CreditEnv(20), 'PPO_CreditEnv_20', True)