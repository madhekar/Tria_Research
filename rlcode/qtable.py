import numpy as np
import gym
from gym import Env

import tria_rl
#env = gym.make('tria_rl/TriaClimate-v0')
env_id = 'tria_rl/TriaClimate-v0'
# Create the environment
env = gym.make(env_id)

# Set the learning rate and discount factor
alpha = 0.9
gamma = 0.9

# Initialize the Q-table
q_table = np.zeros((3, 3, 5))

# Set the number of episodes
num_episodes = 100_000

# Set the exploration rate
exploration_rate = 1.0

# Set the exploration decay rate
exploration_decay_rate = 0.001

# Run the iterations
for i in range(num_episodes):
    # Reset the environment
    state = env.reset()
    reward = 0

    # Run the maximum number of time steps
    for t in range(10):
        # Choose an action
        if np.random.uniform(0, 1) > exploration_rate:
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()
        
            print(action)

        # Take a step in the environment
        next_state, reward, done, info = env.step(action)
        #print(observation)

        # Update the Q-table
        q_table[state[0], state[1], action] = (1 - alpha) * q_table[state[0], state[1], action] + alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1]]))

        # Reset the state
        state = next_state
        # Check if the episode has ended
        if done:
            # Decrease the exploration rate
            exploration_rate = exploration_rate * (1 - exploration_decay_rate)
            #print("exploration_rate =", exploration_rate)
            break
