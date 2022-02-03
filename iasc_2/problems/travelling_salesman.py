import random as rnd
import math
from search_algorithms.simulated_anealing.simulated_anealing_problem import SimulatedAnealingProblem
from search_algorithms.genetic.genetic_problem import GeneticProblem
from search_algorithms.hill_climbing.hill_climbing_problem import HillClimbingProblem
import matplotlib.pyplot as plt


class TravellingSalesman(HillClimbingProblem, SimulatedAnealingProblem, GeneticProblem):
    def __init__(self, n_cities=16, world_size=128):
        """
        'world_size': World size (both vertically and horizontally).
        'n_cities': Number of cities.
        Example with world_size=128 and n_cities=4:
        [[28, 106], [41, 23], [121, 72], [127, 16]]
        """
        self.n_cities = n_cities
        self.world_size = world_size

    def initial_state(self):
        """
        Returns state (world) with the following format (size=128 and 
        n_cities=4, for example): [[28, 106], [41, 23], [121, 72], [127, 16]]\n
        """
        self.n_cities = self.n_cities
        self.world_size = self.world_size
        state = []

        for _ in range(self.n_cities):
            # Generate random city:
            city = [rnd.randint(0, self.world_size),
                    rnd.randint(0, self.world_size)]
            # Keep generating if it already exists:
            while city in state:
                city = [rnd.randint(0, self.world_size),
                        rnd.randint(0, self.world_size)]
            state.append(city)
        return state

    def shuffle(self, state):
        new_state = state.copy()
        rnd.shuffle(new_state)
        return new_state

    def best_neighbour(self, state):
        """
        Takes a 'state' and swaps two, and only two cities in order to ensure the
        minimum total distance traveled.\n Returns the new state.
        """
        # Ridiculously large value:
        best_value = self.world_size*self.world_size*self.n_cities*-1
        # List that will contain all best neighbours:
        best_neighbours = []

        # Loop through all cities (except last one):
        for i in range(self.n_cities - 1):
            # Loop through all cities (from i + 1):
            for j in range(i + 1, self.n_cities):
                # Get state with swapped cities:
                new_state = self.swap_cities(state.copy(), i, j)
                # Get value of state with swapped cities:
                new_value = self.value(new_state)
                if new_value > best_value:
                    # A new best value was found:
                    best_value = new_value
                    # Clear best neighbours, and append this one:
                    best_neighbours.clear()
                    best_neighbours.append(new_state.copy())
                elif new_value == best_value:
                    # Append neighbour:
                    best_neighbours.append(new_state.copy())

        # Randomly choose from best neighbours found:
        return best_neighbours[rnd.randint(0, len(best_neighbours) - 1)]

    def random_neighbour(self, state):
        """
        Takes a 'state' and moves one, and only one queen randomly.\n Returns
        the new state.
        """
        # Randomize cities to swap:
        c1 = rnd.randint(0, self.n_cities - 1)
        c2 = rnd.randint(0, self.n_cities - 1)
        # Enssure c2 != c1:
        while c2 == c1:
            c2 = rnd.randint(0, self.n_cities - 1)

        # Return state with swapped cities:
        return self.swap_cities(state.copy(), c1, c2)

    def value(self, state):
        """
        Takes a 'state' and returns the total distance traveled.
        """
        distance = 0
        for i in range(self.n_cities):
            distance += self.distance_between_cities(
                state[i], state[0 if i + 1 >= self.n_cities else i + 1])
        return distance*-1

    def population(self, size):
        population = []
        state = self.initial_state()
        for _ in range(size):
            rnd.shuffle(state)
            population.append(state.copy())
        return population

    def crossover(self, element1, element2):
        crossover_point = rnd.randint(
            1, self.n_cities-2)
        new_gene = element1[:crossover_point]
        for gene in element2:
            if gene not in new_gene:  # Prevent duplicates
                new_gene.append(gene)
        return new_gene

    def mutate(self, element):
        # Swap two chromosomes
        new_gene = element.copy()
        i = rnd.randint(0, self.n_cities - 1)
        j = rnd.randint(0, self.n_cities - 1)
        while j == i:
            j = rnd.randint(0, self.n_cities - 1)
        new_gene[i], new_gene[j] = new_gene[j], new_gene[i]
        return new_gene

    def fitness(self, element):
        print(element)
        return math.exp(self.value(element)*(2/(self.n_cities*self.world_size)))

    def plot(self, algorithm, initial_state, initial_distance, final_state, final_distance):
        # Append first city to the end, so salesman ends where he started:
        initial_state.append(initial_state[0])
        final_state.append(final_state[0])
        # initial_state = np.reshape(initial_state, (length + 1, 2))
        # final_state = np.reshape(final_state, (length + 1, 2))

        # Plot:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
        fig.canvas.manager.set_window_title(
            "Travelling Salesman (" + algorithm + ")")
        ax[0].set_box_aspect(1)
        ax[1].set_box_aspect(1)
        ax[0].set_title(
            str(round(abs(initial_distance), 1)) + " kms traveled!")
        ax[1].set_title(str(round(abs(final_distance), 1)) + " kms traveled!")
        ax[0].plot(*zip(*initial_state), 'bo-')
        ax[1].plot(*zip(*final_state), 'bo-')
        for i in range(len(initial_state)-1):
            ax[0].annotate(f'  {i+1}', (initial_state[i]
                           [0], initial_state[i][1]))
        for i in range(len(final_state)-1):
            ax[1].annotate(f'  {i+1}', (final_state[i][0], final_state[i][1]))

        plt.show()

    def swap_cities(self, cities, c1, c2):
        """
        Swap 'c1' with 'c2' in list cities.
        """
        cities[c1], cities[c2] = cities[c2], cities[c1]
        return cities

    def distance_between_cities(self, c1, c2):
        """
        Calculate distance between 'c1' and 'c2'.
        """
        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
