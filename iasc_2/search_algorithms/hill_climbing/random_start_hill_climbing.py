from search_algorithms.hill_climbing.stochastic_hill_climbing import StochasticHillClimbing
from search_algorithms.search_algorithm import SearchAlgorithm


class RandomStartHillClimbing(SearchAlgorithm):
    """
    'problem': Problem. Should extend 'HillClimbingProblem'
    'n_iterations_without_change': Number of successive iterations with
    the same value necessary to abort search.
    """

    def __init__(self, problem, n_iterations=16, n_iterations_without_change=24):
        self.problem = problem
        self.n_iterations_without_change = n_iterations_without_change
        self.n_iterations = n_iterations

    def search(self):
        searcher = StochasticHillClimbing(
            self.problem, n_iterations_without_change=self.n_iterations_without_change)
        searcher.search()
        self.initial_state = searcher.initial_state
        self.initial_value = searcher.initial_value
        best_state = searcher.final_state
        best_value = searcher.final_value

        for iteration in range(self.n_iterations):
            searcher.initial_state = self.problem.shuffle(
                searcher.initial_state)
            searcher.search()
            new_state = searcher.final_state
            new_value = searcher.final_value

            print('Iteration %s > state: %s, value: %.2f' %
                  (str(iteration).zfill(len(str(self.n_iterations))), new_state, new_value))

            if new_value < best_value:
                best_state = new_state.copy()
                best_value = new_value

        self.final_state = best_state
        self.final_value = best_value
