import gymnasium as gym
import d4rl

env = gym.make("antmaze-large-diverse-v2", render_mode='rgb_array', disable_env_checker=True)
print(env.reset())
env.step(env.action_space.sample())
env.render()
d4rl.qlearning_dataset(env)