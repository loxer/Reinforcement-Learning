from checkers import *
from createLog import *
import numpy as np
import random
import time
#from IPython.display import clear_output

env = Checkers(4)

action_space_size = env.action_space()
state_space_size = env.state_space()

q_table = np.zeros((state_space_size, action_space_size))

# num_episodes = 100000
num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

start_exploration_rate = exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.0000001
exploration_decay_rate = 0.00001

rewards_all_episodes = []
timestamp = time.gmtime()

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0
 
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
        
        # Add new reward
        rewards_current_episode += reward

        if done == True: 
            break

    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    
    # Add current episode reward to total rewards list
    rewards_all_episodes.append(rewards_current_episode)


# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/10000)
count = 10000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/10000)))
    count += 10000





# From here the logging starts
gameInformation = env.getLoggingInformation()
timeFormat = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) # thx to Metalshark: https://stackoverflow.com/questions/3220284/how-to-customize-the-time-format-for-python-logging

simulationInformation = [timeFormat, action_space_size, state_space_size, q_table, num_episodes, max_steps_per_episode, learning_rate, 
                         discount_rate, exploration_rate, exploration_decay_rate, max_exploration_rate, min_exploration_rate, start_exploration_rate]

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
extra_notes
numberOfFields          x
numberOfplayers         
stonesPlayer1           x
action_space_size       y
state_space_size        y
q_table_size            w
q_table (itself)        y
num_episodes            y
max_steps_per_episode   y
learning_rate           y
discount_rate           y
exploration_rate        y
max_exploration_rate    y
min_exploration_rate    y
exploration_decay_rate  y

reward_invalid_step     x
reward_valid_step       x
reward_loss             x
reward_win              x
rewards_per_thousand_episodes
success_rate_per_thousand_episodes
success_rate_overall_episodes


list at what episode the agent won
 """
