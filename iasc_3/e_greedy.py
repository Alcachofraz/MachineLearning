import random as rnd
from sparse_memory import *


class ActionSelection:
    def select_action(self, state: State) -> Action:
        pass


class EGreedy(ActionSelection):
    def __init__(self, memory: LearnMemory, actions: list, epsilon: float):
        self.memory = memory
        self.actions = actions
        self.epsilon = epsilon

    def max_action(self, state: State) -> Action:
        rnd.shuffle(self.actions)
        return max(self.actions, key=lambda a: self.memory.Q(state, a))

    def benefit(self, state: State) -> Action:
        return self.max_action(state)

    def explore(self) -> Action:
        return self.actions[rnd.randint(0, len(self.actions) - 1)]

    def select_action(self, state: State) -> Action:
        if rnd.random() > self.epsilon:
            return self.benefit(state)
        else:
            return self.explore()
