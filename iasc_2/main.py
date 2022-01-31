from problems.nqueens import NQueens
from problems.travelling_salesman import TravellingSalesman
from search_algorithms.hill_climbing.stochastic_hill_climbing import StochasticHillClimbing
from search_algorithms.hill_climbing.random_start_hill_climbing import RandomStartHillClimbing
from search_algorithms.simulated_anealing.simulated_anealing import SimulatedAnealing
from search_algorithms.genetic.genetic import Genetic
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def plot_n_queens(initial_state, initial_collisions, final_state, final_collisions):
    # Add 1 to all elements:
    initial_state = initial_state[:] = [
        number + 1 for number in initial_state]
    final_state = final_state[:] = [
        number + 1 for number in final_state]

    # Build boards:
    initial_board = [[1 if j + 1 == initial_state[i]
                      else 0 for j in range(len(initial_state))] for i in range(len(initial_state))]
    final_board = [[1 if j + 1 == final_state[i]
                    else 0 for j in range(len(final_state))] for i in range(len(final_state))]

    # Plots:
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.canvas.manager.set_window_title("Nqueens")
    ax[0].set_title(str(initial_collisions) + " Collisions!")
    ax[1].set_title(str(final_collisions) + " Collisions!")
    ax[0].imshow(initial_board, cmap=colors.ListedColormap(['black', 'white']))
    ax[1].imshow(final_board, cmap=colors.ListedColormap(['black', 'white']))
    ax[0].set_xticks(np.arange(0, len(initial_state), 1))
    ax[0].set_yticks(np.arange(0, len(initial_state), 1))
    ax[1].set_xticks(np.arange(0, len(initial_state), 1))
    ax[1].set_yticks(np.arange(0, len(initial_state), 1))
    ax[0].grid(color="w")
    ax[1].grid(color="w")
    plt.show()


def plot_travelling_salesman(initial_state, initial_distance, final_state, final_distance):
    # Append first city to the end, so salesman ends where he started:
    initial_state.append(initial_state[0])
    final_state.append(final_state[0])
    # initial_state = np.reshape(initial_state, (length + 1, 2))
    # final_state = np.reshape(final_state, (length + 1, 2))

    # Plot:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    fig.canvas.manager.set_window_title("Travelling Salesman")
    ax[0].set_box_aspect(1)
    ax[1].set_box_aspect(1)
    ax[0].set_title(str(round(initial_distance, 1)) + " kms traveled!")
    ax[1].set_title(str(round(final_distance, 1)) + " kms traveled!")
    ax[0].plot(*zip(*initial_state), 'bo-')
    ax[1].plot(*zip(*final_state), 'bo-')
    for i in range(len(initial_state)-1):
        ax[0].annotate(f'  {i+1}', (initial_state[i][0], initial_state[i][1]))
    for i in range(len(final_state)-1):
        ax[1].annotate(f'  {i+1}', (final_state[i][0], final_state[i][1]))

    plt.show()


def nqueens(search, n, ):
    """
    'search': stochastic_hill_climbing, random_start_hill_climbing, simulated_anealing, genetic
    'n': Number of queens and, consequently, number of rows and columns of the board.
    """
    problem = NQueens(n_queens=n)
    if (search == "stochastic_hill_climbing"):
        searcher = StochasticHillClimbing(problem=problem, n_iterations_without_change=24)
    elif (search == "random_start_hill_climbing"):
        searcher = RandomStartHillClimbing(problem=problem, n_iterations=16, n_iterations_without_change=24)
    elif (search == "simulated_anealing"):
        searcher = SimulatedAnealing(problem=problem)
    elif (search == "genetic"):
        searcher = Genetic(problem=problem, n_iterations_without_change=24)
    else:
        print("Invalid searcher. Choose from the following:\nstochastic_hill_climbing\nrandom_start_hill_climbing\nsimulated_anealing\ngenetic")
        return
    searcher.search()
    plot_n_queens(searcher.initial_state, searcher.initial_value,
                  searcher.final_state, searcher.final_value)


def travelling_salesman(search, n, size):
    """
    'search': stochastic_hill_climbing, random_start_hill_climbing, simulated_anealing, genetic
    'n': Number of cities
    'size': World size (world is a square)
    """
    problem = TravellingSalesman(n_cities=n, world_size=size)
    if (search == "stochastic_hill_climbing"):
        searcher = StochasticHillClimbing(problem=problem, n_iterations_without_change=24)
    elif (search == "random_start_hill_climbing"):
        searcher = RandomStartHillClimbing(problem=problem, n_iterations=16, n_iterations_without_change=24)
    elif (search == "simulated_anealing"):
        searcher = SimulatedAnealing(problem=problem)
    elif (search == "genetic"):
        searcher = Genetic(problem=problem)
    else:
        print("Invalid searcher. Choose from the following:\nstochastic_hill_climbing\nrandom_start_hill_climbing\nsimulated_anealing\ngenetic")
    searcher.search()
    plot_travelling_salesman(searcher.initial_state, searcher.initial_value,
                             searcher.final_state, searcher.final_value, )


#nqueens(search="stochastic_hill_climbing", n=8)
#travelling_salesman(search="stochastic_hill_climbing", n=8, size=100)
#nqueens(search="random_start_hill_climbing", n=8)
#travelling_salesman(search="random_start_hill_climbing", n=16, size=100) 
nqueens(search="simulated_anealing", n=8)
#travelling_salesman(search="simulated_anealing", n=16, size=100)