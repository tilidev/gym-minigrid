# Gym-Minigrid - `package_only` branch

This branch is reduced to the necessary structure and files to be used as a submodule for other projects.

## Quickstart

### Python version and environment

The recommended Python version for this fork is 3.10.5. Gym-Minigrid generally expects Python 3.5+.
Gym-Minigrid is designed with minimal dependencies in mind. Required packages are especially OpenAI Gym and Numpy. Make sure to use Gym version 0.21.0, as newer versions introduce breaking changes for which Gym-Minigrid is not adjusted.

- Python 3.10.5
- Gym 0.21.0
- Numpy (1.22.4)
- Matplotlib (3.5.1; optional, needed for display)

packages installation:

```sh
pip install gym==0.21.0
conda install matplotlib==3.5.1
```

Numpy is automatically installed with gym, ready to go!

### Setting up `gym_minigrid` in your project

If you simply want to use the RiskyPathEnvironment for your project, I recommend to initialize this repository as a submodule in your current project structure.

Code for setup, assuming project is already a git repository:

```sh
git submodule add -b package_only https://github.com/tilidev/gym_minigrid
git commit -m "add submodule gym_minigrid"
```

### Does everything work as expected?

If you followed all previous steps, the following script should run without issue in your current project.

```python
from time import sleep

import gym
import gym_minigrid
from gym_minigrid.envs import RiskyPathEnv

env = gym.make("MiniGrid-RiskyPath-v0")

num_eps = 5

for i in range(num_eps):
	obs = env.reset()
	env.render("human", tile_size=32)
	sleep(0.5)
	done = False
	while not done:
		obs, reward, done, info = env.step(env.action_space.sample())
		print(f"reward: {reward}")
		env.render("human", tile_size=32)
		sleep(0.5)
		if done:
			print(f"Episode {i} terminated")
```

### RiskyPathEnv configuration

Keyword arguments to add to `gym.make()` method:

Please note that the environment is built in such a way that default lava positions are automatically generated. Use lists of custom positions to change this behavior. Keep in mind 

- `width` (int, optional): Environment width with walls. Defaults to 11.
- `height` (int, optional): Environment height with walls. Defaults to 11.
- `show_agent_dir` (bool, optional): Whether or not to show the agent's current intended direction. Defaults to False.
- `agent_start_pos` (tuple, optional): Tuple to indicate starting position. Top left corner is (0,0). Defaults to (2,9).
- `goal_positions` (list, optional): List of positions for goal tiles. Defaults to [(1,3)].
- `lava_positions` (list, optional): Custom list of lava tile positions. Defaults to None.
- `spiky_active` (bool, optional): Whether or not to activate spiky tiles. Defaults to False.
- `spiky_positions` (list, optional): List of positions for spiky tiles. Defaults to None.
- `reward_spec` (dict, optional): Specification of rewards. For default configuration, see below.
- `slip_proba` (float, optional): Probability of agent slipping. Should be in interval [0,1] Defaults to 0.
- `wall_rebound` (bool, optional): Whether or not to rebound from walls if agent collides with them. Defaults to False.
- `max_steps` (int, optional): Number of maximal steps until automatic environment reset. Defaults to 150.
- `seed` (int, optional): Random seed. Defaults to 1337.

Reward dict must have the following structure:

```python
default_rewards = {
    "step_penalty" : 0,
    "goal_reward" : 1,
    "absorbing_states" : False,
    "absorbing_reward_goal" : 0,
    "absorbing_reward_lava" : -1,
    "risky_tile_reward" : 0,
    "lava_reward" : -1
}
```

### Wrappers

See gym_minigrid docs for wrappers. The wrappers are all saved in `gym_minigrid/wrappers.py`. In gym_minigrid, the default environment observation space is a python dictionary. To change this behavior to boxed observation space, I recommend using the followin wrappers.

- Use `TensorObsWrapper` to create a low-dimensional (fully observable) tensor-encoded env representation (observation space: spaces.Box)
- Use `ImgObsWrapper` on `RGBImgObsWrapper` on env to return only an rgb image in form of a numpy array. The tile pixel resolution can be set in the init method of `RGBImgObsWrapper`.
