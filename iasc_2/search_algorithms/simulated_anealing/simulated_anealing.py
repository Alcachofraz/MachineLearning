
import random as rnd
import math
from search_algorithms.search_algorithm import SearchAlgorithm


class SimulatedAnealing(SearchAlgorithm):
    def __init__(self, problem, n_iterations):
        self.problem = problem
        self.n_iterations = n_iterations

    def search(self):
        current_state = self.problem.initial_state()
        current_value = self.problem.value(current_state)
        best_state, best_value = current_state.copy(), current_value
        self.initial_state, self.initial_value = current_state.copy(), current_value

        for iteration in range(self.n_iterations):
            neighbour_state = self.problem.random_neighbour(current_state)
            neighbour_value = self.problem.value(neighbour_state)

            if neighbour_value < best_value:
                best_state, best_value = neighbour_state.copy(), neighbour_value
                print('Iteration %s > state: %s, value: %.2f' %
                      (str(iteration).zfill(len(str(self.n_iterations))), best_state, best_value))

            delta_e = current_value - neighbour_value
            T = self.n_iterations / float(iteration + 1)

            if delta_e < 0 or rnd.random() <= math.exp(-delta_e/T):
                current_state, current_value = neighbour_state.copy(), neighbour_value

        self.final_state, self.final_value = best_state.copy(), best_value
