from catboost.datasets import amazon
import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from related_work.RL_AAAI_2018.env.AmazonEmployeeEnv import AmazonEmployeeEvn
from catboost.datasets import amazon
from scipy.stats import pearsonr
from sklearn import preprocessing
from scipy import sparse
import random
from related_work.RL_AAAI_2018.agent.agent import DQNAgent
import torch


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)#为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)#为所有GPU设置随机种子
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed = 777
    np.random.seed(seed)
    seed_torch(seed)

    env = AmazonEmployeeEvn(9)
    num_frames = 50000
    memory_size = 1000
    batch_size = 64
    target_update = 200
    epsilon_decay = 1 / 10000

    agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)

    agent.train(num_frames)
