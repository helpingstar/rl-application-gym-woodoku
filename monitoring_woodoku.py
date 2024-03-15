import gym_woodoku
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch
from gym_woodoku.wrappers import ObservationMode, TerminateIllegalWoodoku, RewardMode
from gymnasium.wrappers import DtypeObservation, TransformObservation, TransformReward, ReshapeObservation, RecordVideo
from gymnasium.spaces import Box
from torch.distributions.categorical import Categorical


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(device))
        return -p_log_p.sum(-1)


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(n_channel, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 1024),
            nn.ReLU(),
        )
        self.actor = nn.Linear(1024, env.action_space.n)
        self.critic = nn.Linear(1024, 1)

    def get_action(self, x, invalid_action_mask=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = CategoricalMasked(logits=logits, masks=invalid_action_mask)
        action = probs.sample()
        return action


n_channel = 4
monitoring_name = "test1"
record_interval = 1
render_mode = "rgb_array"
model_path = "weight/channel4/cleanrl_woodoku_ppo_v2_action_mask_244100.pt"
n_episode = 30

if render_mode == "human":
    env = gym.make("gym_woodoku/Woodoku-v0", render_mode="human")
else:
    env = gym.make("gym_woodoku/Woodoku-v0", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        f"videos/{monitoring_name}",
        episode_trigger=lambda x: True,
        disable_logger=False,
        fps=5,
    )
env = RewardMode(env, "woodoku")
env = ObservationMode(env, n_channel=n_channel)
env = gym.wrappers.ReshapeObservation(env, (1, n_channel, 9, 9))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(env).to(device)
agent.load_state_dict(torch.load(model_path))

obs, info = env.reset()
with torch.inference_mode():
    while n_episode > 0:
        obs = torch.Tensor(obs).to(device)
        action = agent.get_action(obs, invalid_action_mask=torch.Tensor(info["action_mask"]).to(device))
        action = action.item()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            n_episode -= 1
            if n_episode > 0:
                obs, info = env.reset()
env.close()
