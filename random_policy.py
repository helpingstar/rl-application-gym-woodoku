import gymnasium as gym
import numpy as np
import gym_woodoku
from gym_woodoku.wrappers import ObservationMode, TerminateIllegalWoodoku

env = gym.make('gym_woodoku/Woodoku-v0', game_mode='woodoku', render_mode='human')
env = TerminateIllegalWoodoku(env, -5)
env = ObservationMode(env)
# env = gym.wrappers.RecordVideo(env, video_folder='./video_folder')

observation, info = env.reset()
for i in range(100000):
    # action = np.random.choice(np.where(info['action_mask'])[0])

    action = env.action_space.sample()

    obs, reward, terminated, _, info = env.step(action)
    print(reward, terminated)
    print()
    if terminated:
        observation, info = env.reset()
env.close()