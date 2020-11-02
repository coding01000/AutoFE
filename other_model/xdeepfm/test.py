import sys
import os

sys.path.append(os.path.abspath('.'))
import environment

print(environment.HousePriceEnv(10).base_score)
