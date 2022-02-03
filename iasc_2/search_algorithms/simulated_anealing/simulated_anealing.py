
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
        t = 0
        T0 = self.n_iterations
        T = T0

        while True:
            neighbour_state = self.problem.random_neighbour(current_state)
            neighbour_value = self.problem.value(neighbour_state)

            if neighbour_value > best_value:
                best_state, best_value = neighbour_state.copy(), neighbour_value
                print('Iteration %s > state: %s, value: %.2f' %
                      (str(t).zfill(4), best_state, best_value))

            delta_e = neighbour_value - current_value
            T = T0 * 0.99 ** t

            if T == 0:
                break

            if delta_e > 0 or rnd.random() <= math.exp(delta_e/T):
                current_state, current_value = neighbour_state.copy(), neighbour_value

            t += 1

        self.final_state, self.final_value = best_state.copy(), best_value
