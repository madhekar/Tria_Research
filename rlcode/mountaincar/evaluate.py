import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
import os
import time

# Parallel environments
env = make_vec_env("MountainCarContinuous-v0", n_envs=1)

logdir = f"logs/Mountain-{time.time()}"
# The learning agent and hyperparameters
model = PPO(
    policy=MlpPolicy,
    env=env,
    seed=0,
    batch_size=32,
    ent_coef=0.00429,
    learning_rate=7.77e-05,
    n_epochs=10,
    n_steps=8,
    gae_lambda=0.9,
    gamma=0.9999,
    clip_range=0.1,
    max_grad_norm =5,
    vf_coef=0.19,
    use_sde=True,
    policy_kwargs=dict(log_std_init=-3.29, ortho_init=False),
    verbose=1,
    tensorboard_log=logdir
    )

models_dir = "models/Mountain-1683931011.646336"
model_path = f"{models_dir}/140000"
best_model = PPO.load(model_path, env=env)
obs = env.reset()
while True:
    action, _states = best_model.predict(obs)
    print('action: ', action, '_states: ', _states)
    obs, rewards, dones, info = env.step(action)
    print('obs: ', obs, ' rewards: ', rewards)
    env.render()  #use Python IDE to check, I havn't figure out how to render in Notebook
