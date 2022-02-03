from search_algorithms.search_algorithm import SearchAlgorithm
import random as rnd


class Genetic(SearchAlgorithm):
    def __init__(self, problem, n_iterations=128, population_size=48, mutation_rate=0.1):
        self.problem = problem
        self.n_iterations = n_iterations
        self.n_genes = population_size
        self.mutation_rate = mutation_rate

    def roulette(self, population):
        roulette = []
        for chromosome in population:
            for _ in range(round(self.problem.fitness(chromosome)*len(population))):
                roulette.append(chromosome)
        return roulette[rnd.randint(0, len(roulette) - 1)]

    def search(self):
        population = self.problem.population(self.n_genes)
        self.initial_state = population[0]
        self.initial_value = self.problem.value(self.initial_state)
        for _ in range(self.n_iterations):
            new_population = []  # Empty set
            for _ in range(0, len(population)):
                x = self.roulette(population)
                y = self.roulette(population)

                child = self.problem.crossover(x, y)

                if rnd.random() < self.mutation_rate:
                    child = self.problem.mutate(child)

                new_population.append(child)
            population = new_population
        rnd.shuffle(population)
        self.final_state = max(population, key=self.problem.fitness)
        self.final_value = self.problem.value(self.final_state)
