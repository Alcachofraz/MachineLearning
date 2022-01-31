from search_algorithms.search_algorithm import SearchAlgorithm
import random as rnd
import math


class SimulatedAnealing(SearchAlgorithm):
    def __init__(self, problem):
        self.problem = problem

    def search(self):
        current_state = self.problem.initial_state()
        current_value = self.problem.value(current_state)
        self.initial_state = current_state
        self.initial_value = current_value
        t = 0
        while True:
            t += 1
            T = self.schedule(t)

            if T <= 0:
                self.final_state = current_state
                self.final_value = current_value
                return

            neighbour_state = self.problem.random_neighbour(current_state)
            neighbour_value = self.problem.value(current_state)

            delta_e = current_value - neighbour_value
            
            if delta_e > 0 or rnd.random() <= math.exp(delta_e/T):
                current_state = neighbour_state.copy()
                current_value = neighbour_value

    def schedule(self, time):
        T0 = 100
        alpha = 0.99
        return T0 * alpha ** time
