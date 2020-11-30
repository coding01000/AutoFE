# -*- coding: utf-8 -*-
import os
import torch
from utils.PPO import Memory, PPO

device = torch.device("cpu")  # "cuda:0" if torch.cuda.is_available() else "cpu"


def run(env, env_name, load=False):
    ############## Hyperparameters ##############
    env_name = env_name
    # creating environment
    env = env
    state_dim = env.state_dim
    action_dim = env.action.action_dim
    render = False
    log_interval = 200  # print avg reward in the interval 200
    max_episodes = 500000  # max training episodes
    max_timesteps = 300  # max timesteps in one episode
    n_latent_var = 64  # number of variables in hidden layer
    update_timestep = 2000  # update policy every n timesteps 2000
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None
    max_score = 0
    max_ep = []
    #############################################

    path = os.path.abspath(__file__).split('AutoFE')[0] + 'AutoFE'
    path = os.path.join(path, 'model')
    path = os.path.join(path, f'{env_name}.pth')
    print(path)
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    if load:
        print(f'load: {path}')
        ppo.policy.load_state_dict(torch.load(path))
        ppo.policy_old.load_state_dict(torch.load(path))

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
                max_score = max(max_score, reward)
                max_ep = env.one_episode
                print(i_episode, '---------max_score:', max_score, '------------', t, '---------', reward)
                break

        avg_length += t

        # logging
        if i_episode % log_interval == 0:
            avg_length = (avg_length / log_interval)
            running_reward = ((running_reward / log_interval))
            torch.save(ppo.policy.state_dict(), path)
            # torch.save(i_episode, './model/PPO_{}.pth'.format('nn'))
            # break
            os.system('clear')
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            print(max_ep)
            running_reward = 0
            avg_length = 0


def replay(env, env_name):
    state_dim = env.state_dim
    action_dim = env.action.action_dim
    render = False
    max_timesteps = 500
    n_latent_var = 64  # number of variables in hidden layer
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    #############################################

    n_episodes = 3
    max_timesteps = 300

    path = os.path.abspath(__file__).split('AutoFE')[0] + 'AutoFE'
    path = os.path.join(path, 'model')
    path = os.path.join(path, f'{env_name}.pth')
    print(path)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    ppo.policy_old.load_state_dict(torch.load(path))

    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = ppo.policy_old.act(state, memory)
            state, reward, done = env.step(action)
            ep_reward += reward
            if done:
                break

        print(f'episode: {ep}, reward: {ep_reward}, generator: {env.one_episode}')
