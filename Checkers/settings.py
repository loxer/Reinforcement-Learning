from game import *
from trainer import *
from commander import *

# Training Settings
board_size = 4
num_trainings = 1
num_episodes_per_training = 10000
max_steps_per_episode = 100                # not used in this game, since every step is either an advancement towards the goal or invalid and gaming ending

learning_rate = 0.1
discount_rate = 0.99

start_exploration_rate = 1
max_exploration_rate = start_exploration_rate
min_exploration_rate = "0"                  # as string for logs, will be casted to float at the training
exploration_decay_rate = "0.001"            # same

# Algorithm Settings
use_classic_algorithm = False               # updates q_table after every step
use_advanced_algorithm = True               # updates q_table after every episode backwards for all steps

# Reward Settings
reward_valid_step = 0
reward_milestone = 0
reward_won = 1
reward_lost = 0

# Logging Settings
log_notes = ""
statistics_separation_counter = int(num_episodes_per_training / 10)
create_log_file = False

# Saving Settings
log_save_path = "Checkers\\Logs\\Version_"
agent_save_path = "Checkers\\agents.hdf5"

# Console Settings
max_chars_of_explanations_per_line = 91
max_chars_to_explanations = 20

# Storing Settings in Lists
algorithm_settings = [use_classic_algorithm, use_advanced_algorithm]
training_settings = [board_size, num_trainings, num_episodes_per_training, max_steps_per_episode, learning_rate, discount_rate, 
                       start_exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate, algorithm_settings]

reward_settings = [reward_valid_step, reward_milestone, reward_won, reward_lost]

logging_settings = [log_notes, statistics_separation_counter, log_save_path, create_log_file]

console_settings = [max_chars_of_explanations_per_line, max_chars_to_explanations]


# Create a board
board = Game(board_size, reward_settings)

# Create the trainer
trainer = Trainer(board, training_settings, logging_settings)

# Start the programm
commander = Commander(trainer, board_size)
commander.start(board, console_settings, agent_save_path)