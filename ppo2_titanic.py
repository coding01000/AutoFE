# -*- coding: utf-8 -*-

import numpy as np
import scipy
import os
from itertools import combinations, permutations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_validate
import pandas as pd
import torch
from PPO import PPO, Memory


class Action(object):
    def __init__(self, data, bounds):
        self.data = data
        self.feature_nums = data.shape[1]
        self.action_num = 2  # 处理方式数目
        self.combinations = list(combinations([i for i in range(self.feature_nums)], 2))  # 特征组合
        self.action_dim = self.feature_nums * self.action_num + len(self.combinations)
        self.actions = [i for i in range(self.action_dim)]  # 动态搜索空间
        # self.selected = [] #所有已经处理过的action
        self.processed = {}  # 处理过后数据的存储
        self.episode_done = []  # 单次episode做过的action
        self.bounds = bounds
        self.ptr = 0

    def reset(self):
        self.ptr = 0
        self.actions = [i for i in range(self.action_dim)]
        self.episode_done = []

    def step(self, _action):
        action = np.zeros(2, dtype='int')

        is_combinations = _action / self.action_num >= self.feature_nums
        if not is_combinations:
            action[0] = _action / self.action_num
            action[1] = _action % self.action_num
        else:
            action[0] = self.combinations[_action - self.feature_nums * self.action_num - 1][0]
            action[1] = self.combinations[_action - self.feature_nums * self.action_num - 1][1]

        print(is_combinations, '..............', _action, '---------', f'[{action[0]}, {action[1]}]')

        if _action not in self.actions:
            return True

        self.episode_done.append(_action)

        if _action not in self.processed:
            if not is_combinations:
                column_name = self.data.columns[action[0]]
                if action[1] == 0:
                    tmp = (self.data[[column_name]] + 1).apply(np.log)
                    tmp = scipy.sparse.csr_matrix(tmp)
                elif action[1] == 1:
                    ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')
                    tmp = ohe.fit_transform(self.data[[column_name]])
            else:
                c1 = self.data.columns[action[0]]
                c2 = self.data.columns[action[1]]
                tmp = self.data[c1].apply(str) + self.data[c2].apply(str)
                ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')
                tmp = ohe.fit_transform(tmp.values.reshape(-1, 1))
            self.processed[_action] = tmp
        if is_combinations:
            self.actions.remove(_action)
        else:
            start = action[0] * self.action_num
            end = start + self.action_num
            for i in range(start, end):
                self.actions.remove(i)
        return self.isDone()

    def isDone(self):
        self.ptr += 1
        if self.ptr >= self.bounds:
            return True
        else:
            return False

    def data_sets(self):
        x = scipy.sparse.hstack([self.processed[i] for i in self.episode_done])
        return x


class TitanicEnv(object):
    def __init__(self, bounds):
        # data, _ = amazon()
        data = pd.read_csv('./dataset/Titanic/train.csv')
        self.label = data['Survived'].values
        data.drop(['PassengerId', 'Survived', 'Ticket', 'Name', 'Cabin'], axis=1, inplace=True)

        data['Embarked'] = data['Embarked'].astype(str)
        data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])
        data['Sex'] = LabelEncoder().fit_transform(data['Sex'])

        data = data.fillna(0)

        self.action = Action(data, bounds)
        self.base_score = self.auc_score(data.values)
        self.state_dim = self.action.action_dim
        self.state = np.zeros(self.state_dim)
        self.one_episode = []  # 存储action执行序列
        self.episodes = {}

    def reset(self):
        self.state = np.zeros(self.state_dim)
        self.one_episode = []
        self.action.reset()
        return self.state

    def step(self, _action):
        self.one_episode.append(_action)
        done = self.action.step(_action)
        if done:
            reward = self._reward()
        else:
            reward = 0
        self.state[_action] += 1
        return self.state, reward, done

    def _reward(self):
        if self.action.ptr < self.action.bounds:
            return -10
        one_episode = "".join(map(str, self.one_episode))
        if one_episode in self.episodes:
            return self.episodes[one_episode]
        self.episodes[one_episode] = self.auc_score(self.action.data_sets()) - self.base_score
        return self.episodes[one_episode]

    def auc_score(self, x):
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            fit_intercept=True,
            random_state=432,
            solver='liblinear',
            max_iter=1000,
            # n_jobs=-1,
        )
        y = self.label
        stats = cross_validate(model, x, y, groups=None, scoring='accuracy',
                               cv=5, return_train_score=True)
        return stats['test_score'].mean() * 100


def main():
    ############## Hyperparameters ##############
    env_name = "Titanic_Acc"
    # creating environment
    env = TitanicEnv(15)
    state_dim = env.state_dim
    action_dim = env.action.action_dim
    render = False
    solved_reward = 300  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 500000  # max training episodes
    max_timesteps = 300  # max timesteps in one episode
    n_latent_var = 64  # number of variables in hidden layer
    update_timestep = 2000  # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    ppo.policy.load_state_dict(torch.load('./model/PPO_Titanic_Acc.pth'))
    ppo.policy_old.load_state_dict(torch.load('./model/PPO_Titanic_Acc.pth'))

    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # #........
    # cnt = 0
    # #........

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, done = env.step(action)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                print(i_episode, '---------------------', t, '---------', reward)
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        # if running_reward > (log_interval * solved_reward):
        #     print("########## Solved! ##########")
        #     torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
        #     break

        # logging
        if i_episode % log_interval == 0:
            avg_length = (avg_length / log_interval)
            running_reward = ((running_reward / log_interval))
            torch.save(ppo.policy.state_dict(), './model/PPO_{}.pth'.format(env_name))
            torch.save(i_episode, './model/PPO_{}.pth'.format('nn'))
            os.system('clear')
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


main()
