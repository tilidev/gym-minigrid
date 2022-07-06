import gym

# import gym_minigrid to include registered environments 
import gym_minigrid

from PIL import Image
from gym_minigrid.envs.risky import RiskyPathEnv

env = gym.make("MiniGrid-RiskyPath-v3")

obs = env.reset()

done = False

action_dict = {i.value : i.name for i in RiskyPathEnv.Actions}

for i in range(50):
    action = env.action_space.sample()
    assert action in [0,1,2,3]

    obs, reward, done, info = env.step(action)

    print(
        f"step={env.step_count}",
        f"action={action_dict[action]}",
        f"reward={reward}",
        f"done={done}"
    )

    if done:
        obs = env.reset()
        print("reset environment\n")


# img = Image.fromarray(obs)
# img.show("Observation of time step 12 as an rgb-image")