from state_action import *


class Wavefront:
    def __init__(self, gain: int = 10):
        self.gain = gain
        self.path = []

    def propagate(self, world):
        V = {}
        wavefront = []
        # Gama proportional to world:
        gama = len(world.world)/(len(world.world)+1)

        V[world.target] = self.gain
        wavefront.append(world.target)

        while len(wavefront) > 0:  # While wavefront still contains states:
            s = wavefront.pop(0)
            for a in self.adjacents(world.world, s):
                v = V[s] * gama  # Atenuate
                # In case there's a worse value adjacent, ieplace and add to wavefront
                if v > V.get(a, -1):
                    V[a] = v
                    wavefront.append(a)
        return V

    def adjacents(self, world, s: State) -> list[State]:
        # Check for adjacent position (vertically and horizontally only):
        adjacentes: list[State] = []
        if s.x > 0:
            e = State(s.x - 1, s.y)
            if world[e.y][e.x] != -1:
                adjacentes.append(e)
        if s.x < len(world[0]) - 1:
            e = State(s.x + 1, s.y)
            if world[e.y][e.x] != -1:
                adjacentes.append(e)
        if s.y > 0:
            e = State(s.x, s.y - 1)
            if world[e.y][e.x] != -1:
                adjacentes.append(e)
        if s.y < len(world) - 1:
            e = State(s.x, s.y + 1)
            if world[e.y][e.x] != -1:
                adjacentes.append(e)
        return adjacentes

    def get_path(self, world, V):
        # Retorna os estados no caminho do estado inicial até o alvo
        state = world.state
        self.path = []
        while state != world.target:
            self.path.append(state)
            next_state = max(self.adjacents(world.world, state),
                             key=lambda s: V[s])
            state = next_state
        return self.path
