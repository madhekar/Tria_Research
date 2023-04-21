from research_pap_model_torch import Agent
import gym
import numpy as np 

env = gym.make('LunarLanderContinuous-v2')

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
                  batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

score_history = []
np.random.seed(0)
for i in range(1000):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs= new_state
            
        score_history.append(score)
        print('e ', i, 'score %.2f' % score, '100 game avg %.2f' %np.mean(score_history[-100:]))

        if i % 25 == 0:
            agent.save_models()