from humanPlayer import *
import numpy as np

learning_rate = 0.1
discount_rate = 0.99

learning_attributes = [learning_rate, discount_rate]

match = HumanPlayer(4)
match.start(0, learning_attributes)