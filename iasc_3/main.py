from agent import *
from world import *
from state_action import *


actions = [
    Action(1, 0),  # Right
    Action(-1, 0),  # Left
    Action(0, -1),  # Up
    Action(0, 1)  # Down
]

# Algorithm [choose one]:
#agent = DynaQAgent(actions=actions)
agent = WavefrontAgent(actions=actions)

# Create world:
world = World("iasc_3/worlds/world1.txt", algorithm_name=type(agent).__name__)

# Process challenge:
agent.process(world)
