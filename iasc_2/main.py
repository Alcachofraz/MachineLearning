from problems.nqueens import NQueens
from search_algorithms.stochastic_hill_climbing import StochasticHillClimbing
from search_algorithms.random_start_hill_climbing import RandomStartHillClimbing
from search_algorithms.simulated_anealing import SimulatedAnealing
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def plot_n_queens(state, collisions):
    # Add 1 to all elements:
    state = state[:] = [
        number + 1 for number in state]

    # Build board:
    board = [[1 if j + 1 == state[i]
              else 0 for j in range(len(state))] for i in range(len(state))]

    # Plot:
    plt.figure()
    plt.title("NQueens » " + str(collisions) + " Collisions!")
    plt.imshow(board, cmap=colors.ListedColormap(['black', 'white']))
    plt.grid(color="w")
    plt.show()


searcher = StochasticHillClimbing(
    problem=NQueens(n_queens=8), n_iterations_without_change=24)
searcher.search()
plot_n_queens(searcher.final_state, searcher.final_value)
