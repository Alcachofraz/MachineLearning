from state_action import *


class LearnMemory:
    def update(state: State, action: Action, q: float):
        pass

    def Q(state: State, action: Action) -> float:
        pass


class SparseMemory(LearnMemory):
    def __init__(self, default: float = 0):
        self.default = default
        self.memory = {}

    def Q(self, state: State, action: Action) -> float:
        return self.memory.get((state, action), self.default)

    def update(self, state: State, action: Action, q: float):
        self.memory[(state, action)] = q
