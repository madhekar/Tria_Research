import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

class MyEnv(gym.Env):
    """
    A simple environment with a grid of cells that the agent can navigate.
    The agent can move up, down, left, or right. Each movement earns or
    loses a fixed amount of reward. The episode ends when the agent reaches
    a designated goal cell or a designated trap cell.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Define the action and observation space
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(low=0, 
                                            high=8, 
                                            shape=(3,3))  # 3x3 grid

        # Initialize the grid
        self.grid = np.zeros((3,3))
        self.grid[0,0] = 1  # start cell
        self.grid[2,2] = 2  # goal cell
        self.grid[1,1] = -1  # trap cell

        # Initialize the state
        self.state = (0,0)
        self.steps_taken = 0

        # Seed the random number generator
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Take a step in the environment.
        """
        # Update the state based on the action
        if action == 0:  # up
            self.state = (self.state[0] - 1, self.state[1])
        elif action == 1:  # down
            self.state = (self.state[0] + 1, self.state[1])
        elif action == 2:  # left
            self.state = (self.state[0], self.state[1] - 1)
        elif action == 3:  # right
            self.state = (self.state[0], self.state[1] + 1)

        # Check if the state is out of bounds
        if self.state[0] < 0:
            self.state = (self.state[0]+1 , self.state[1])
            
        elif self.state[0] > 2:
            self.state = (self.state[0]-1 , self.state[1])
        
        elif self.state[1] < 0:
            self.state = (self.state[0] , self.state[1]+1)
        
        elif self.state[1] > 2:
            self.state = (self.state[0] , self.state[1]-1)

        # Increment the number of steps taken
        self.steps_taken += 1

        # Check if the episode is finished
        done = False
        if self.grid[self.state[0], self.state[1]] == 2:
            reward = 1
            done = True
        elif self.grid[self.state[0], self.state[1]] == -1:
            reward = -1
            done = True
        else:
            reward = 0

        # Return the observation, reward, and whether the episode is finished
        return self.state, reward, done, {}

    def reset(self):
        """
        Reset the environment.
        """
        # Set the agent position to the start cell
        self.state = (0,0) 
        # Reset the step counter
        self.steps_taken = 0 
        return self.state 
    
    def render(self, mode='human'):
        """
        Render the current state of the environment.
        """
        grid = self.grid.copy()
        # Mark the agent's position
        grid[self.state[0], self.state[1]] = 8  
        print(grid)

# Create the environment
env = MyEnv()

# Set the learning rate and discount factor
alpha = 0.9
gamma = 0.9

# Initialize the Q-table
q_table = np.zeros((3, 3, 4))

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
        
            # print(action)

        # Take a step in the environment
        next_state, reward, done, info = env.step(action)
        #print(observation)

        # Update the Q-table
        q_table[state[0], state[1], action] = (1 - alpha) * q_table[state[0], state[1], action] +\
                                               alpha * (reward +\
                                                gamma * np.max(q_table[next_state[0], next_state[1]]))

        # Reset the state
        state = next_state
        # Check if the episode has ended
        if done:
            # Decrease the exploration rate
            exploration_rate = exploration_rate * (1 - exploration_decay_rate)
            #print("exploration_rate =", exploration_rate)
            break

print(q_table)

print(np.argmax(q_table, axis=2))
