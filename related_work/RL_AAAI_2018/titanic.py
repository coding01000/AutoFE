import sys
sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
sys.path.append('/home/zhangyanfeng/AutoFE')
# sys.path.append('/Users/mac/PycharmProjects/AutoFE')
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
from related_work.RL_AAAI_2018.env.TitanicEnv import TitanicEnv
from related_work.RL_AAAI_2018.agent.agent import DQNAgent
import warnings
from utils import init_seed
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    init_seed.init_seed()

    env = TitanicEnv(5)
    load_name = 'Titanic'

    num_frames = 50000
    memory_size = 1000
    batch_size = 64
    target_update = 200
    epsilon_decay = 1 / 10000

    agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay, load_name, True)

    agent.train(num_frames)
