from e_greedy import *
from model_tr import *


class ReinforcedLearning:
    def __init__(self, memory: LearnMemory, action_selection: ActionSelection, alpha: float, gama: float):
        self.memory = memory
        self.action_selection = action_selection
        self.alpha = alpha
        self.gama = gama

    def learn(state: State, action: Action, r: float, next_state: State, next_action: Action = None):
        pass


class QLearning(ReinforcedLearning):
    def learn(self, state: State, action: Action, r: float, next_state: State):
        next_action = self.action_selection.max_action(next_state)
        q_state_action = self.memory.Q(state, action)
        q_nest_state_next_action = self.memory.Q(
            next_state, next_action)
        q = q_state_action + self.alpha * \
            (r + self.gama * q_nest_state_next_action - q_state_action)
        self.memory.update(state, action, q)


class DynaQ(QLearning):
    def __init__(self, memory: LearnMemory, action_selection: ActionSelection, alpha: float, gama: float, sim_num: int):
        super().__init__(memory, action_selection, alpha, gama)
        self.sim_num = sim_num
        self.model = ModelTR()

    def simulate(self):
        for i in range(self.sim_num):
            state, action, r, next_state = self.model.sample()
            super().learn(state, action, r, next_state)

    def learn(self, state: State, action: Action, r: float, next_state: State):
        super().learn(state, action, r, next_state)
        self.model.update(state, action, r, next_state)
        self.simulate()
