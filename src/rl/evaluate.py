from stable_baselines3 import DQN
from src.environment.mahjong_env import MahjongEnv

model = DQN.load("models/mahjong_dqn_agent")

env = MahjongEnv

for episode in range(10):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, = env.step(action)
        total_reward += reward