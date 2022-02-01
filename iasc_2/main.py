from problems.nqueens import NQueens
from problems.travelling_salesman import TravellingSalesman
from search_algorithms.hill_climbing.stochastic_hill_climbing import StochasticHillClimbing
from search_algorithms.hill_climbing.random_start_hill_climbing import RandomStartHillClimbing
from search_algorithms.simulated_anealing.simulated_anealing import SimulatedAnealing
from search_algorithms.genetic.genetic import Genetic
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def search(algorithm, problem, n_iterations=16, n_iterations_without_change=24):
    """
    'search': stochastic_hill_climbing, random_start_hill_climbing, simulated_anealing, genetic
    'n': Number of queens and, consequently, number of rows and columns of the board.
    """
    problem = problem
    if (algorithm == "stochastic_hill_climbing"):
        searcher = StochasticHillClimbing(
            problem=problem, n_iterations_without_change=n_iterations_without_change)
        a = "Stochastic Hill Climbing"
    elif (algorithm == "random_start_hill_climbing"):
        searcher = RandomStartHillClimbing(
            problem=problem, n_iterations=n_iterations, n_iterations_without_change=n_iterations_without_change)
        a = "Random Start Hill Climbing"
    elif (algorithm == "simulated_anealing"):
        searcher = SimulatedAnealing(
            problem=problem, n_iterations=n_iterations)
        a = "Simulated Anealing"
    elif (algorithm == "genetic"):
        searcher = Genetic(problem=problem)
        a = "Genetic"
    else:
        print("Invalid searcher. Choose from the following:\nstochastic_hill_climbing\nrandom_start_hill_climbing\nsimulated_anealing\ngenetic")
        return
    searcher.search()
    problem.plot(a, searcher.initial_state, searcher.initial_value,
                 searcher.final_state, searcher.final_value)


#search(algorithm="stochastic_hill_climbing", problem=NQueens(n_queens=8), n_iterations_without_change=24)
#search(algorithm="stochastic_hill_climbing", problem=TravellingSalesman(n_cities=16, world_size=100), n_iterations_without_change=24)
search(algorithm="random_start_hill_climbing", problem=NQueens(
    n_queens=8), n_iterations=16, n_iterations_without_change=24)
search(algorithm="random_start_hill_climbing",
       problem=TravellingSalesman(n_cities=16, world_size=100), n_iterations=16, n_iterations_without_change=24)
#search(algorithm="simulated_anealing", problem=NQueens(n_queens=8), n_iterations=16)
#search(algorithm="simulated_anealing", problem=TravellingSalesman(n_cities=16, world_size=100), n_iterations=16)
#search(algorithm="genetic", problem=NQueens(n_queens=8))
#search(algorithm="genetic", problem=TravellingSalesman(n_cities=16, world_size=100))
