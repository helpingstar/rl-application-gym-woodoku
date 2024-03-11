import gym_woodoku
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch
from gym_woodoku.wrappers import ObservationMode, TerminateIllegalWoodoku, RewardMode
from gymnasium.wrappers import DtypeObservation, TransformObservation, TransformReward, ReshapeObservation, RecordVideo
from gymnasium.spaces import Box
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


n_channel = 4
monitoring_name = "test1"
record_interval = 1
render_mode="human"
model_path = (
    "runs/gym_woodoku/Woodoku-v0__woodoku_ppo_v2_action_mask__1__1709902965/cleanrl_woodoku_ppo_v2_action_mask_170870.pt"
)

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
            layer_init(nn.Conv2d(n_channel, 32, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 5 * 5, 1024)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(1024, env.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(1024, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None, invalid_action_mask=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        # probs = Categorical(logits=logits)
        probs = CategoricalMasked(logits=logits, masks=invalid_action_mask)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if render_mode == "human":
    env = gym.make("gym_woodoku/Woodoku-v0", render_mode="human")
else:
    env = gym.make("gym_woodoku/Woodoku-v0", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, f"videos/{monitoring_name}", episode_trigger=lambda x: (x % record_interval == 0)
    )
env = RewardMode(env, "woodoku")
env = gym.wrappers.TransformReward(env, lambda r: r * 0.01)
env = ObservationMode(env, n_channel=n_channel)
env = gym.wrappers.ReshapeObservation(env, (1, n_channel, 9, 9))
env = gym.wrappers.RecordEpisodeStatistics(env)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(env).to(device)
agent.load_state_dict(torch.load(model_path))

obs, info = env.reset()
with torch.inference_mode():
    for i in range(100000):
        obs = torch.Tensor(obs).to(device)
        action = agent.get_action_and_value(obs, invalid_action_mask=torch.Tensor(info["action_mask"]).to(device))[0]
        action = action.item()
        # action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
env.close()