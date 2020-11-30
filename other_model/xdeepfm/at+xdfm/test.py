import os
import sys

sys.path.append(os.path.abspath('.'))

import environment as env
from dataprocessing import dataset
import numpy as np

if __name__ == "__main__":
    bike = env.BikeShareEnv(15)
    data, a = dataset.get_bike_share(False)
    # print(data.shape)
    # print(np.isnan(data).any())
    a, is_combinations = bike.action.replay(0)
    # print(a.shape)
    # print(np.isnan(a).any())
    # print(a)
    # data['a'], b = bike.action.replay(2)
    # print(np.isnan(data['a']).any())
