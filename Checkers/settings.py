from game import *
from commander import *

# Training Settings
board_size = 4
num_trainings = 1
num_episodes_per_training = 30000
max_steps_per_episode = 1000                # not used in this game, since every step is either an advancement towards the goal or invalid and gaming ending

learning_rate = 1
discount_rate = 0.99

start_exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = "0.0000000"          # as string for logs, will be casted to float at the training
exploration_decay_rate = "0.0001165"        # same

# Reward Settings
reward_valid_step = 1
reward_milestone = 1
reward_won = 1
reward_lost = -1

# Logging Settings
log_notes = "Test specific agent"
statistics_separation_counter = 5000
create_log_file = False

# Saving Settings
log_save_path = "Checkers\\Logs\\Version_"
agent_save_path = "Checkers\\agents.hdf5"

# Console Settings
max_chars_of_explanations_per_line = 91
max_chars_to_explanations = 20

# Storing Settings in Lists
training_settings = [board_size, num_trainings, num_episodes_per_training, max_steps_per_episode, learning_rate, discount_rate, 
                       start_exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate]

reward_settings = [reward_valid_step, reward_milestone, reward_won, reward_lost]

logging_settings = [log_notes, statistics_separation_counter, log_save_path, create_log_file]

console_settings = [max_chars_of_explanations_per_line, max_chars_to_explanations]


# Create a board
board = Game(board_size, reward_settings)


# Start the programm
commander = Commander(board_size)
commander.start(board, training_settings, logging_settings, console_settings, agent_save_path)