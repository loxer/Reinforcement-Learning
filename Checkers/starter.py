from game import *
from humanPlayer import *

# Training Settings
board_size = 6
num_simulations = 1
num_episodes_per_simulation = 30000
max_steps_per_episode = 1000

learning_rate = 0.1
discount_rate = 0.99

start_exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = "0.0000000"          # as string for logs, will be casted to float at the training
exploration_decay_rate = "0.0000065"        # same

# Reward Settings
reward_valid_step = 0
reward_milestone = 0
reward_won = 0
reward_lost = -1

# Logging Settings
log_notes = "No more min_exploration_rate and Negative Rewards (for losing), only"
statistics_separation_counter = 2500

# Saving Settings
log_save_path = "Checkers\\Logs\\Version_"
agent_save_path = "Checkers\\agents.hdf5"

# Storing Settings in Lists
simulation_settings = [board_size, num_simulations, num_episodes_per_simulation, max_steps_per_episode, learning_rate, discount_rate, 
                       start_exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate]

reward_settings = [reward_valid_step, reward_milestone, reward_won, reward_lost]

logging_settings = [log_notes, statistics_separation_counter, log_save_path]


# Create a board
board = Game(board_size, reward_settings)


# Start the programm
controller = HumanPlayer(board_size)
controller.start(board, simulation_settings, logging_settings, agent_save_path)