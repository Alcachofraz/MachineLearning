from search_algorithms.search_algorithm import SearchAlgorithm


class StochasticHillClimbing(SearchAlgorithm):
    final_state = None
    final_value = None

    """
    'problem': Problem. Should extend 'HillClimbingProblem'
    'n_iterations_without_change': Number of successive iterations with
    the same value necessary to abort search.
    """

    def __init__(self, problem, n_iterations_without_change=24):
        self.problem = problem
        self.n_iterations_without_change = n_iterations_without_change

    def search(self):
        current_state = None
        current_value = None
        previous_state = self.problem.initial_state()
        iterations_without_change = 0

        while True:
            # Get current neighbour:
            current_state = self.problem.best_neighbour(previous_state)

            # Get values of previous and current neighbour:
            current_value = self.problem.value(current_state)
            previous_value = self.problem.value(previous_state)

            # Compare values and increment iterations_without_change if due:
            if current_value == previous_value:
                iterations_without_change += 1
            else:
                iterations_without_change = 0

            if previous_value < current_value or iterations_without_change > self.n_iterations_without_change:
                self.final_state = previous_state
                self.final_value = previous_value
                return

            previous_state = current_state.copy()
