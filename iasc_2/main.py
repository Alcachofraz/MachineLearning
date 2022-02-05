from problems.nqueens import NQueens
from problems.travelling_salesman import TravellingSalesman
from search_algorithms.hill_climbing.stochastic_hill_climbing import StochasticHillClimbing
from search_algorithms.hill_climbing.random_start_hill_climbing import RandomStartHillClimbing
from search_algorithms.simulated_anealing.simulated_anealing import SimulatedAnealing
from search_algorithms.genetic.genetic import Genetic
import pygad
import numpy as np


def search(algorithm, problem, n_iterations=16, n_iterations_without_change=24, population_size=48, mutation_rate=0.1):
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
        searcher = Genetic(problem=problem, population_size=population_size,
                           n_iterations=n_iterations, mutation_rate=mutation_rate)
        a = "Genetic"
    else:
        print("Invalid searcher. Choose from the following:\nstochastic_hill_climbing\nrandom_start_hill_climbing\nsimulated_anealing\ngenetic")
        return
    searcher.search()
    problem.plot(a, searcher.initial_state, searcher.initial_value,
                 searcher.final_state, searcher.final_value)


def search_builtin(problem, n_generations=100, allow_duplicate_genes=True, gene_type=int, population_size=100):
    population = problem.population(population_size)
    def fitness_function(element, index): return problem.fitness(element)
    ga_instance = pygad.GA(
        fitness_func=fitness_function,
        num_generations=n_generations,
        num_parents_mating=2,
        gene_type=gene_type,
        initial_population=population,
        parent_selection_type="rws",
        crossover_type="single_point",
        crossover_probability=1,
        mutation_type="swap",
        mutation_probability=0.1,
        allow_duplicate_genes=allow_duplicate_genes,
        stop_criteria="reach_1")
    ga_instance.run()
    solution = ga_instance.best_solution()[0]
    problem.plot("PyGad", population[0], problem.value(
        population[0]), solution, problem.value(solution))


#search(algorithm="stochastic_hill_climbing", problem=NQueens(n_queens=8), n_iterations_without_change=24)
#search(algorithm="stochastic_hill_climbing", problem=TravellingSalesman(n_cities=64, world_size=100), n_iterations_without_change=24)
#search(algorithm="random_start_hill_climbing", problem=NQueens(n_queens=32), n_iterations=16, n_iterations_without_change=24)
#search(algorithm="random_start_hill_climbing", problem=TravellingSalesman(n_cities=64, world_size=100), n_iterations=16, n_iterations_without_change=24)
#search(algorithm="simulated_anealing", problem=NQueens(n_queens=32), n_iterations=100)
#search(algorithm="simulated_anealing", problem=TravellingSalesman(n_cities=64, world_size=100), n_iterations=16)
#search(algorithm="genetic", problem=NQueens(n_queens=32), n_iterations=48, population_size=100, mutation_rate=0.1)
search(algorithm="genetic", problem=TravellingSalesman(
    n_cities=8, world_size=100), n_iterations=48, population_size=48, mutation_rate=0.1)

#search_builtin(problem=NQueens(n_queens=16), n_generations=200, allow_duplicate_genes=False, gene_type=int, population_size=200)
# Para o travelling salesman não funciona porque a biblioteca só aceita doubles ou integers. Ter-se-ia de alterar complemamente o
# problema ou arranjar uma forma de representar um tuplo num único número.
