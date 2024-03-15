import gymnasium as gym
from gym_woodoku.wrappers import ObservationMode, TerminateIllegalWoodoku, RewardMode, AddStraight


env = gym.make("gym_woodoku/Woodoku-v0", render_mode="human")
env = RewardMode(env, "woodoku")
env = TerminateIllegalWoodoku(env, 0)
env = ObservationMode(env, n_channel=4)
env = AddStraight(env)
env = gym.wrappers.RecordEpisodeStatistics(env)

obs, info = env.reset()
for i in range(100000):
    action = 0
    obs, reward, terminated, _, info = env.step(action)
    print(obs)
    if terminated:
        obs, info = env.reset()
env.close()
