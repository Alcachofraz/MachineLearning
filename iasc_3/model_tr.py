import random as rnd
from state_action import *


class ModelTR:
    def __init__(self):
        self.T = {}
        self.R = {}

    def update(self, state: State, action: Action, r: float, next_state: State):
        self.T[(state, action)] = next_state
        self.R[(state, action)] = r

    def sample(self):
        state, action = list(self.T)[rnd.randint(0, len(self.T.keys()) - 1)]
        next_state = self.T[(state, action)]
        r = self.R[(state, action)]
        return state, action, r, next_state
