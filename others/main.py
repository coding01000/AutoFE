# -*- coding: utf-8 -*-
import sys

sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
import os

import torch
from utils.PPO import Memory, PPO
from environment.BikeShareEnv import BikeShareEnv
from utils import init_seed

device = torch.device("cpu")  # "cuda:0" if torch.cuda.is_available() else "cpu"


def main(env, env_name, load=False):
    ############## Hyperparameters ##############
    env_name = env_name
    # creating environment
    env = env
    state_dim = env.state_dim
    action_dim = env.action.action_dim
    render = False
    log_interval = 200  # print avg reward in the interval
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

    if load:
        ppo.policy.load_state_dict(torch.load(f'./model/{env_name}.pth'))
        ppo.policy_old.load_state_dict(torch.load(f'./model/{env_name}.pth'))

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
            torch.save(ppo.policy.state_dict(), './model/{}.pth'.format(env_name))
            # torch.save(i_episode, './model/PPO_{}.pth'.format('nn'))
            os.system('clear')
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    init_seed.init_seed()
    # main(BikeShareEnv(15), 'PPO_BikeShare_15', False)
    # main(TitanicEnv(15), 'PPO_Titanic_15', False)
    print(BikeShareEnv(15).base_score)
    # print(HousePriceEnv(80).base_score)
    # print( HousePriceEnv(80).base_score)
