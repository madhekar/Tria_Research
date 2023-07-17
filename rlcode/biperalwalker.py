import random
import numpy as np
from scoop import futures
import gym


def do(it):
    env = gym.make("BipedalWalker-v3")
    random.seed(it)
    print(it)
    np.random.seed(it)
    env.seed(it)
    env.action_space.seed(it)
    env.reset()
    observations = []
    for i in range(3):
        while True:
            action = env.action_space.sample()
            ob, reward, done, _ = env.step(action)
            observations.append(ob)
            if done:
                break
    return observations


if __name__ == "__main__":
    maxit = 20
    results1 = futures.map(do, range(2, maxit))
    results2 = futures.map(do, range(2, maxit))
    for a,b in zip(results1, results2):
        if np.array_equiv(a, b):
            print("equal, yay")
        else:
            print("not equal :(")