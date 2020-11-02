import sys
sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
from environment.AmazonEnv import AmazonEnv
from utils.init_seed import init_seed


if __name__ == '__main__':
    init_seed()
    env = AmazonEnv(15)
    base_score = env.base_score
    print(f'base score: {base_score}')
    max_iter = 1000
    max_round = 10
    max_score = 0
    score = 0
    for k in range(max_round):
        max_score = 0
        for i in range(max_iter):
            action = env.sample()
            _, reward, done = env.step(action)
            max_score = max(max_score, reward)
            if done:
                print(reward)
                env.reset()
        score += max_score
    print(base_score + score / max_round)
