
### Import OpenAI Gym framework, including a number of environment examples.
import gym
from myEnv.snakeEnv import SnakeEnv

### Import Stable-Baselines3 for RL algorithm implementations
from stable_baselines3 import PPO






# Create environment
env = SnakeEnv(render_mode=None)

# Create RL agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./logs/")  #, replay_buffer_class=HerReplayBuffer)

# Train RL agent
model.learn(total_timesteps=100000000, tb_log_name="PPO_100M_tsteps")

# Visualize trained RL agent
env = SnakeEnv(render_mode="human")
obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    #env.render()
    if done:
      obs = env.reset()