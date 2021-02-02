from checkers import *
from createLog import *
import numpy as np
import random
import time
#from IPython.display import clear_output

log_notes = ""
COUNT = count = 10000            # in what steps the statistics are printed

env = Checkers(4)
action_space_size = env.action_space()
state_space_size = env.state_space()
q_table = np.zeros((state_space_size, action_space_size))

num_episodes = 100000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

start_exploration_rate = exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = log_min_exploration_rate = "0.0000001"
exploration_decay_rate = log_exploration_decay_rate = "0.00001"

min_exploration_rate = float(min_exploration_rate)
exploration_decay_rate = float(exploration_decay_rate)

rewards_all_episodes = []
valid_steps_all_episodes = []
milestones_all_episodes = []
wins_of_all_episodes = []

timestamp = time.gmtime()

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0
    valid_steps_current_episode = 0
    milestones_current_episodes = 0
 
    # initialize new episode params
    for step in range(max_steps_per_episode):
        
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space_sample()
            
        # Take new action
        new_state, reward, done, info = env.step(action)
        
        # Update Q-table
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        
        # Set new state
        state = new_state
        
        # Save data for statistics
        rewards_current_episode += reward
        
        if info[0] == True:
            valid_steps_current_episode += 1
            if info[1] == True:
                milestones_current_episodes += 1

        # Check if episode finished
        if done == True:
            if info[2] == False:
                wins_of_all_episodes.append(0)
            else:
                wins_of_all_episodes.append(1)
            break

    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    
    # Add current episode reward to total rewards list
    rewards_all_episodes.append(rewards_current_episode)
    valid_steps_all_episodes.append(valid_steps_current_episode)
    milestones_all_episodes.append(milestones_current_episodes)

rewards_all_episodes = np.split(np.array(rewards_all_episodes),num_episodes/COUNT)
valid_steps_all_episodes = np.split(np.array(valid_steps_all_episodes),num_episodes/COUNT)
milestones_all_episodes = np.split(np.array(milestones_all_episodes),num_episodes/COUNT)
wins_of_all_episodes = np.split(np.array(wins_of_all_episodes),num_episodes/COUNT)
# success_statistics = np.array([rewards_all_episodes, valid_steps_all_episodes, milestones_all_episodes, wins_of_all_episodes])

log_reward_statistics = ""

for episode_phase in range(num_episodes // COUNT):
    rewards_all_episodes[episode_phase]

    log_reward_statistics += str(count) + ": "
    log_reward_statistics += "Rewards: " + str(sum(rewards_all_episodes[episode_phase]/COUNT)) + " || Valid Steps: " + str(sum(valid_steps_all_episodes[episode_phase]))
    log_reward_statistics += " || Milestones: " + str(sum(milestones_all_episodes[episode_phase])) + " || Wins: " + str(sum(wins_of_all_episodes[episode_phase]))
    log_reward_statistics += "\n"
    count += COUNT


# print(success_statistics[3, 1])

# for test in success_statistics[3]:
#     log_reward_statistics += str(count) + ": " + str(sum(test)) + "\n"
#     count += COUNT

# print("********Average reward per thousand episodes********\n")
# for r in rewards_all_episodes:
#     log_reward_statistics += str(count) + ": " + str(sum(r/COUNT)) + "\n"
#     count += COUNT


# for r in rewards_per_thousand_episodes:
    # log_reward_statistics += str(count) + ": "
    # log_reward_statistics += "Rewards: " + str(sum(r/COUNT)) + "       Valid Steps: " + str(sum(r))
    # log_reward_statistics += "\n"
    # count += COUNT


# From here the logging starts
gameInformation = env.getLoggingInformation()
timeFormat = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) # thx to Metalshark: https://stackoverflow.com/questions/3220284/how-to-customize-the-time-format-for-python-logging

simulationInformation = [timeFormat, action_space_size, state_space_size, q_table, num_episodes, max_steps_per_episode, learning_rate, discount_rate, 
                        exploration_rate, log_exploration_decay_rate, max_exploration_rate, log_min_exploration_rate, start_exploration_rate, log_notes,
                        log_reward_statistics]

logger = CreateLog(gameInformation, simulationInformation)
logMessage = logger.getLog()

version = gameInformation[0]
FILE = "Logs\\Version_" + str(version) + "\\" + timeFormat + "_123.txt"
logFile = open(FILE,"w+")
logFile.write(logMessage)
print(logMessage)


# List of values for the logs
""" timestemp           x
version_of_game         x
extra_notes             x
numberOfFields          x
numberOfplayers         /
stonesPlayer1           x
action_space_size       x
state_space_size        x
q_table_size            x
q_table (itself)        <-- need to safe it somewhere else
num_episodes            x
max_steps_per_episode   x
learning_rate           x
discount_rate           x
exploration_rate        x
max_exploration_rate    x
min_exploration_rate    x
exploration_decay_rate  x

reward_invalid_step     x
reward_valid_step       x
reward_loss             x
reward_win              x

number of milestones
number of milestones
number of wins
time for process        

rewards_per_thousand_episodes
success_rate_per_thousand_episodes
success_rate_overall_episodes
percentage_of_valid_steps
percentage_of_wins


list at what episode the agent won
 """
