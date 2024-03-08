from woodoku_ppo_v2 import make_env
import gymnasium as gym
from gym_woodoku.wrappers import ObservationMode, TerminateIllegalWoodoku, RewardMode


env = gym.make("gym_woodoku/Woodoku-v0", render_mode="human")
env = RewardMode(env, "woodoku")
env = TerminateIllegalWoodoku(env, 0)
env = ObservationMode(env, n_channel=4)

env = gym.wrappers.RecordEpisodeStatistics(env)

obs, info = env.reset()
for i in range(100000):
    # action = env.action_space.sample(info["action_mask"])
    action = 0
    # print(info["action_mask"], action)
    obs, reward, terminated, _, info = env.step(action)
    print(reward)
    if terminated:
        obs, info = env.reset()
env.close()
