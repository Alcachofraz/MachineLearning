from hill_climbing_problem.hill_climbing_problem import HillClimbingProblem
import random as rnd
import math


class TravellingSalesman(HillClimbingProblem):
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

    def best_neighbour(self, state):
        """
        Takes a 'state' and swaps two, and only two cities in order to ensure the
        minimum total distance traveled.\n Returns the new state.
        """
        # Ridiculously large value:
        best_value = self.world_size*self.world_size*self.n_cities
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
        return best_neighbours[rnd.randint(0, len(best_neighbours))]

    def random_neighbour(self, state):
        """
        Takes a 'state' and moves one, and only one queen randomly.\n Returns
        the new state.
        """
        # Randomize cities to swap:
        c1 = rnd.randint(0, self.n_cities)
        c2 = rnd.randint(0, self.n_cities)
        # Enssure c2 != c1:
        while c2 == c1:
            c2 = rnd.randint(0, self.n_cities)

        # Return state with swapped cities:
        return self.swap_cities(state.copy(), c1, c2)

    def value(self, state):
        """
        Takes a 'state' and returns the total distance traveled.
        """
        distance = 0
        for i in range(self.n_cities):
            distance += self.distance_between_cities(
                state[i], state[i + 1 if i + 1 >= self.n_cities else 0])
        return distance

    def swap_cities(cities, c1, c2):
        """
        Swap 'c1' with 'c2' in list cities.
        """
        cities[c1], cities[c2] = cities[c2], cities[c1]
        return cities

    def distance_between_cities(c1, c2):
        """
        Calculate distance between 'c1' and 'c2'.
        """
        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
