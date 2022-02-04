import numpy as np
import matplotlib.pyplot as plt
from state_action import *


class World:
    def __init__(self, file_name: str, multiplier: float = 10, move_cost: float = 0.01, algorithm_name="Unknown"):
        self.file_name = file_name
        self.world, self.state, self.target = self.load(file_name)
        self.multiplier = multiplier
        self.move_cost = move_cost
        self.movements = 0
        self.turn = 1
        # Create plot with world:
        plt.imshow([[x for x in y] for y in self.world])
        plt.tick_params(axis='x', which='both', bottom=False,
                        top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', right=False,
                        left=False, labelleft=False)
        plt.gcf().canvas.manager.set_window_title(
            '[' + file_name[-10:-4] + '] ' + algorithm_name)

    def load(self) -> list[list[int]]:
        with open(self.file_name, "r") as file:
            lines = file.readlines()
            world: list[list[int]] = np.zeros(
                (len(lines), len(lines[0].removesuffix('\n'))), dtype=int)

            initial_state = State(0, 0)
            target = State(0, 0)

            # Find obstacles [O], initial position [>] and target position [A]:
            m = 0
            for x in lines:
                n = 0
                for y in x[:-1]:
                    if y.__eq__('O'):
                        world[m][n] = -1
                    elif y.__eq__('A'):
                        world[m][n] = 2
                        target = State(n, m)
                    elif y.__eq__('>'):
                        initial_state = State(n, m)
                    n += 1
                m += 1

            return world, initial_state, target

    def regenerate(self):
        self.movements = 0
        self.turn += 1
        self.world, self.state, self.target = self.load(self.file_name)

    def current_state(self) -> State:
        return self.state

    def update_state(self, next_state: State):
        self.state = next_state

    def move(self, action: Action):
        next_state = State(self.state.x + action.dx, self.state.y + action.dy)
        # Increment movements:
        self.movements += 1
        # Next position:
        r = self.world[next_state.y][next_state.x]
        # Retun next state and reinforcement, acording to position and move cost:
        return next_state if r >= 0 else self.state, r * self.multiplier - self.move_cost

    def draw_plot_with_path(self, V, path, world):
        map = [[x for x in y] for y in world]
        for v in V:
            map[v.y][v.x] = V[v]

        for s in path:
            map[s.y][s.x] = -5

        plt.imshow(map)
        plt.show(block=True)

    def draw_plot(self):
        # World:
        position = [[x for x in y] for y in self.world]
        # Set current position:
        position[self.state.y][self.state.x] = 1
        plt.title("Turn: " + str(self.turn) +
                  "\nSteps: " + str(self.movements))
        # Update map plot:
        self.map.set_data(position)
        plt.draw()
        plt.pause(0.001)
