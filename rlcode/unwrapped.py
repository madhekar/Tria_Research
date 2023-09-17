import gym
import numpy as np
env = gym.make("MountainCarContinuous-v0")
env = env.unwrapped # to access the inner functionalities of the class
env.state = np.array([-0.4, 0])
print(env.state)
a, _ = env.reset()
for i in range(500):
    obs, _, _, _ = env.step([1]) # Just taking right in every step   
    print(obs, env.state) #the observation and env.state are same
    env.render()
