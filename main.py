import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from DQN import Agent


def make_env():
    env = gym.make('LunarLander-v2')
    env.reset(seed=0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    return env


def env_info(env):
    print('observation space', env.observation_space)
    print('observation space sample ', env.observation_space.sample())
    print('env.action_space', env.action_space)
    print('env.action_spaces sample', env.action_space.sample())


def interaction(env, agent):
    state = env.reset()
    for j in range(200):
        action = agent.act(state)
        env.render()
        state, reward, done, inf = env.step(action)
        if done:
            break

    env.close()


def DQN(agent, env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint2.pth')
    return scores


def main():
    env = make_env()
    env_info(env)
    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=10)
    # interaction(env, agent)  ## not trained agent
    scores = DQN(agent, env)
    interaction(env,agent)


main()
