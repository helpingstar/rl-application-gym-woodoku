import gymnasium as gym

import gym_woodoku
from gym_woodoku.wrappers import ObservationMode, TerminateIllegalWoodoku, RewardMode, AddStraight
from tqdm import tqdm


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", episode_trigger=lambda x: (x % args.record_interval == 0)
            )
        else:
            env = gym.make(env_id)
        env = RewardMode(env, "woodoku")
        env = gym.wrappers.TransformReward(env, lambda r: r * 0.01)
        env = ObservationMode(env, n_channel=4)
        env = AddStraight(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk


envs = gym.vector.SyncVectorEnv(
    [make_env("gym_woodoku/Woodoku-v0", i, False, "test") for i in range(4)],
)

print(envs.observation_space["board"])

obs, infos = envs.reset()
for i in range(100000):
    action = envs.action_space.sample()
    obs, reward, terminated, truncated, infos = envs.step(action)
    print(obs)
envs.close()
