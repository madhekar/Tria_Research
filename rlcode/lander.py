import gym
env = gym.make("LunarLander-v2") #, render_mode="rgb_array")
observation = env.reset()#seed=42)
for _ in range(10000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, info = env.step(action)
   env.render()

   if terminated:
      observation = env.reset()
env.close()
