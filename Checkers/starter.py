from humanPlayer import *
from simulation import *
import numpy as np

learning_rate = 0.1
discount_rate = 0.99
size = 6
num_simulations = 2

learning_attributes = [learning_rate, discount_rate]

for simulation_episode in range(num_simulations):    
    simulation = Simulation()
    simulation.run(size, simulation_episode + 1)

# match = HumanPlayer(size)
# match.start(0, learning_attributes)