import sys
sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
from related_work.RL_AAAI_2018.env.TitanicEnv import TitanicEnv

env = TitanicEnv(5)
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(7))
print(env.step(2))
env.reset()
print('---------------')
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(7))
print(env.step(2))