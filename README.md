# rl-application-gym-woodoku

This is the code to solve [**gym-woodoku**](https://github.com/helpingstar/gym-woodoku), a reinforcement learning environment based on the woodoku game.

After training, you can modify the `weight_path` in the `monitoring.py` file to record the agent's gameplay.

The parameters for training are inside the `woodoku_ppo_v2*.py` file and can be modified as desired.

The code referenced
* https://github.com/vwxyzjn/cleanrl
* https://github.com/vwxyzjn/invalid-action-masking

![episode_length](/figure/episode_length.png)

* Red: invalid_action_masking
* Bottom blue: invalid_action_penalty
* Top blue: Loaded the last weight of the bottom blue and trained as invalid_action_penalty

**The experimental results including the game's straight will be updated soon.**