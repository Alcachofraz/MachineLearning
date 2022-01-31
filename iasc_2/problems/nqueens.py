import random as rnd
from hill_climbing_problem.hill_climbing_problem import HillClimbingProblem


class NQueens(HillClimbingProblem):
    def __init__(self, n_queens=4):
        self.n_queens = n_queens

    def initial_state(self):
        """
        Returns state (board) with the following format (size=4, for example):\n
        State: [1,2,0,1]\n
        Board: 0100 | 0010 | 1000 | 0100\n
        Note that every queen will only move in their respective row.
        """
        state = []
        # Loop through all rows:
        for _ in range(self.n_queens):
            # Move queen to a random column
            state.append(rnd.randint(0, self.n_queens-1))
        return state

    def best_neighbour(self, state):
        """
        Takes a 'state' and moves one, and only one queen in order to ensure as
        less collisions as possible (both adjacent and diagonal).\n Returns the
        new state.
        """
        # Ridiculously large value:
        best_value = self.n_queens*1000
        # List that will contain all best neighbours:
        best_neighbours = []

        # Loop through all rows (all queens):
        for i in range(self.n_queens):
            # Candidate to new state:
            new_state = state.copy()
            # Current column the queen is in:
            current = state[i]
            # Loop through all columns:
            for j in range(self.n_queens):
                if j != current:  # Except the current one
                    # Move queen:
                    new_state[i] = j
                    # Get value of state with moved queen:
                    new_value = self.value(new_state)
                    if new_value < best_value:
                        # A new best value was found:
                        best_value = new_value
                        # Clear best neighbours, and append this one:
                        best_neighbours.clear()
                        best_neighbours.append(new_state.copy())
                    elif new_value == best_value:
                        # Append neighbour:
                        best_neighbours.append(new_state.copy())

        # Randomly choose from best neighbours found:
        return best_neighbours[rnd.randint(0, len(best_neighbours)-1)]

    def random_neighbour(self, state):
        """
        Takes a 'state' and moves one, and only one queen randomly.\n Returns
        the new state.
        """
        new_state = state.copy()
        # Queen that will be moved (every queen only moves in their row):
        moving_queen = rnd.randint(0, self.n_queens)
        # Set new column to the current one (for now):
        new_column = new_state[moving_queen]
        # Loop until a new column is randomly obtained (that's not the current one):
        while new_column == new_state[moving_queen]:
            new_column = rnd.randint(0, self.n_queens)
        # Move queen:
        new_state[moving_queen] = new_column
        return new_state

    def value(self, state):
        """
        Takes a 'state' and returns the number of collisions.
        """
        collisions = 0
        # Keep in mind that there won't ever be horizontal collisions due to the
        # way this problem is designed.
        # Iterate through all rows (except the last one):
        for i in range(self.n_queens - 1):
            for j in range(i + 1, self.n_queens):
                # Vertical and diagonal collision check
                if state[j] == state[i] or abs((state[i] - state[j])) == abs(i - j):
                    collisions += 1

        # Return collisions:
        return collisions
