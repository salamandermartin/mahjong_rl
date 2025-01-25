from stable_baselines3 import DQN
from stable_baselines.common.env_util import make_vec_env
from srv.environment.mahjong_env import MahjongEnv

env = make_vec_env(MahjongEnv, n_envs = 4)

model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, buffer_size=10000)

model.learn(total_timesteps = 50000)

model.save("models/mahjong_dqn_agent")