# channel 4

## common

```python
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
```

## net1

```python
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(args.n_channel, 32, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 5 * 5, 1024)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(1024, envs.single_action_space.n), std=0.01)
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
```

## net2

```python
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(args.n_channel, 32, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.network2 = nn.Sequential(
            layer_init(nn.Linear(64 * 5 * 5 + 1, 1024)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(1024, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(1024, 1), std=1)

    def pre_get(self, x, straight):
        x = self.network(x)
        x = torch.concat([x, straight / 10], dim=-1)
        x = self.network2(x)
        return x

    def get_value(self, x, straight):
        return self.critic(self.pre_get(x, straight))

    def get_action_and_value(self, x, straight, action=None, invalid_action_mask=None):
        hidden = self.pre_get(x, straight)
        logits = self.actor(hidden)
        # probs = Categorical(logits=logits)
        probs = CategoricalMasked(logits=logits, masks=invalid_action_mask)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
```