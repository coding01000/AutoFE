import sys
import os
sys.path.append(os.path.abspath(__file__).split('AutoFE')[0]+'AutoFE')
from related_work.RL_AAAI_2018.env.CreditEnv import CreditEnv
from related_work.RL_AAAI_2018.agent.agent import DQNAgent
from utils import init_seed
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    init_seed.init_seed()

    env = CreditEnv(5)
    load_name = 'Credit'

    num_frames = 50000
    memory_size = 1000
    batch_size = 64
    target_update = 200
    epsilon_decay = 1 / 10000

    agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay, load_name, False)

    agent.train(num_frames)
# [6, 3, 3, array(3), 7]
