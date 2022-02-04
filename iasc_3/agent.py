from world import World
from dyna_q import DynaQ
from e_greedy import EGreedy
from state_action import Action
from sparse_memory import SparseMemory
from wavefront import Wavefront
import copy


class Agent:
    def __init__(self, actions: list[Action]):
        self.actions = actions

    def process(self, world: World):
        pass


class DynaQAgent(Agent):
    def __init__(self, actions: list[Action]):
        super().__init__(actions)
        self.memory = SparseMemory()
        self.action_selection = EGreedy(
            self.memory, self.actions, 0.1)
        self.learning = DynaQ(
            self.memory, self.action_selection, 0.7, 0.95, 1000)
        self.end = False

    def process(self, world: World):
        while not self.end:
            while not self.end:
                action = self.action_selection.select_action(
                    world.current_state())
                next_state, r = world.move(action)
                self.learning.learn(world.current_state(),
                                    action, r, next_state)
                world.update_state(next_state)
                world.draw_plot()
                if (next_state == world.target):
                    break
            print("Turn: " + str(world.turn).zfill(3) +
                  " | Steps: " + str(world.movements).zfill(4))
            world.regenerate()


class WavefrontAgent(Agent):
    def __init__(self, actions: list[Action]):
        super().__init__(actions)
        self.learning = Wavefront(gain=10)

    def process(self, world: World):
        V = self.learning.propagate(world)
        path = self.learning.get_path(world, V)
        world.draw_plot_with_path(V, path, world.world)
