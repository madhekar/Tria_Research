import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy



import os
import time
from stable_baselines3.common.env_util import make_vec_env
import tria_rl

# Saving logs to visulize in Tensorboard, saving models


models_dir = f"models/Tria-{time.time()}"
logdir = f"logs/Tria-{time.time()}"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Parallel environments
env = make_vec_env('tria_rl/TriaClimate-v0', n_envs=1)
#env = make_vec_env("MountainCarContinuous-v0", n_envs=1)

# The learning agent and hyperparameters
model = PPO(
    policy=MlpPolicy,
    env=env,
    seed=0,
    batch_size=64,
    ent_coef=0.00429,
    learning_rate=7.77e-05,
    n_epochs=100,
    n_steps=8,
    gae_lambda=0.9,
    gamma=0.9999,
    clip_range=0.1,
    max_grad_norm =5,
    vf_coef=0.19,
    use_sde=False,
    policy_kwargs=dict(log_std_init=-3.29, ortho_init=False),
    verbose=1,
    tensorboard_log=logdir
    )



#Training and saving models along the way
TIMESTEPS = 200000
for i in range(5):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/Tria-{TIMESTEPS*i}")

# Check model performance
# load the best model you observed from tensorboard - the one reach the goal/ obtaining highest return
model_path = f"{models_dir}/Tria-600000"
best_model = PPO.load(model_path, env=env)
obs = env.reset()
while True:
    action, _states = best_model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()  #use Python IDE to check, I havn't figure out how to render in Notebook


