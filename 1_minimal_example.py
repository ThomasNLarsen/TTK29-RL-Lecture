
### Import OpenAI Gym framework, including a number of environment examples.
import gym

### Import Stable-Baselines3 for RL algorithm implementations
from stable_baselines3 import A2C


# Create environment
env = gym.make("CartPole-v1")

# Create RL agent
model = A2C("MlpPolicy", env, verbose=1)

# Train RL agent
model.learn(total_timesteps=10_000)

# Visualize trained RL agent
obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()