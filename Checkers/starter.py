from humanPlayer import *
from simulation import *
import numpy as np

learning_rate = 0.1
discount_rate = 0.99
size = 6

learning_attributes = [learning_rate, discount_rate]

simulation = Simulation()
simulation.run(size)

# match = HumanPlayer(size)
# match.start(0, learning_attributes)