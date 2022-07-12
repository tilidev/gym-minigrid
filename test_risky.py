from time import sleep
import gym

# import gym_minigrid to include registered environments 
import gym_minigrid

from PIL import Image
from gym_minigrid.envs.risky import RiskyPathEnv
from gym_minigrid.minigrid import TILE_PIXELS
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, TensorObsWrapper

def run_normal():
    # TODO change method name
    env = gym.make("MiniGrid-RiskyPath-v0", show_agent_dir=False, spiky_active=False)

    obs = env.reset()

    done = False

    action_dict = {i.value : i.name for i in RiskyPathEnv.Actions}

    for i in range(50):
        action = env.action_space.sample()
        assert action in [0,1,2,3]

        tensor_obs = env.tensor_obs()
        obs, reward, done, info = env.step(action)
        print(tensor_obs)
        env.render("human", tile_size=TILE_PIXELS)
        sleep(0.2)

        print(
            f"step={env.step_count}",
            f"action={action_dict[action]}",
            f"reward={reward}",
            f"done={done}"
        )

        if done:
            obs = env.reset()
            print("reset environment\n")

def test_tensor_obs():
    env: RiskyPathEnv = gym.make("MiniGrid-RiskyPath-v0")
    if env.spiky_active:
        expected_shape = (11, 11, 5)
    else:
        expected_shape = (11, 11, 4)
    env = TensorObsWrapper(env)

    num_episodes = 5

    for ep in range(num_episodes):
        obs = env.reset()
        assert obs.shape == expected_shape

        done = False
        cur_step = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            cur_step += 1

            print(f"reward: {reward},     step: {cur_step},     episode: {ep + 1}")
    
    assert obs.shape == expected_shape

def test_full_rgb_obs():
    env = gym.make("MiniGrid-RiskyPath-v0")
    env = RGBImgObsWrapper(env, tile_size=32)
    env = ImgObsWrapper(env)
    # NOTE This is the setup to use when applying DRL algorithms with SB3

    num_episodes = 5

    for ep in range(num_episodes):
        obs = env.reset()

        done = False
        cur_step = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            cur_step += 1
        
    img = Image.fromarray(obs)
    img.show()


# img = Image.fromarray(obs)
# img.show("Observation of time step 12 as an rgb-image")

if __name__ == "__main__":
    # test_tensor_obs()
    # run_normal()
    test_full_rgb_obs()