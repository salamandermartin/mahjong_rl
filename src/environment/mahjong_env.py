import gym
from gym import spaces
import numpy as np

class MahjongEnv(gym.env):
    def __init__(self):
        super(MahjongEnv, self).__init__()

        self.observation_space = spaces.Box(low=0, high=4, shape=(34,), dtype=np.int32)

        self.action_space = spaces.Discrete(34)

        self.reset()

    def reset(self):
        self.hand = np.random.randint(0, 4, size = 34)
        return self.hand
    
    def step(self, action):
        self.hand[action] -= 1
        reward = self._calculate_reward()
        done = self._check_win_condition()
        return self.hand, reward, done, {}
    
    def _calculate_reward(self):
        """Reward system to encourage forming melds"""
        return np.sum(self.hand > 0)
    
    def _check_win_condition(self):
        """Check for a winning hand condition"""
        return np.max(self.hand) == 4  # Example win condition
    
    def render(self, mode='human'):
        """Display the current hand"""
        print("Hand:", self.hand)