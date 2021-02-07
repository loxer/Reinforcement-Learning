from game import *
from logger import *
import numpy as np
import random
import time
import timeit
#from IPython.display import clear_output

class Simulation:
    def __init__(self):
        self.q_table = 0

    def getAgent(self):
        return self.q_table

    def run(self):
        log_notes = ""
        statistics_separation_counter = 50000

        env = Game(4)
        action_space_size = env.action_space()
        state_space_size = env.state_space()
        q_table = np.zeros((state_space_size, action_space_size))

        num_episodes = 300000
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
        invalid_steps_all_episodes = []
        valid_steps_all_episodes = []
        milestones_all_episodes = []
        wins_of_all_episodes = []

        timestamp = time.gmtime()
        timeMeasurement = timeit.default_timer()

        # Q-learning algorithm
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            rewards_current_episode = 0
            valid_steps_current_episode = 0
            invalid_steps_current_episodes = 0
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
                else:
                    invalid_steps_current_episodes += 1

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
            invalid_steps_all_episodes.append(invalid_steps_current_episodes)
            milestones_all_episodes.append(milestones_current_episodes)


        # Preparing statistics for log
        timeMeasurement = timeit.default_timer() - timeMeasurement
        total_valid_steps = sum(valid_steps_all_episodes)
        total_steps = total_valid_steps + sum(invalid_steps_all_episodes)

        rewards_all_episodes = np.split(np.array(rewards_all_episodes),num_episodes/statistics_separation_counter)
        valid_steps_all_episodes = np.split(np.array(valid_steps_all_episodes),num_episodes/statistics_separation_counter)
        invalid_steps_all_episodes = np.split(np.array(invalid_steps_all_episodes),num_episodes/statistics_separation_counter)
        milestones_all_episodes = np.split(np.array(milestones_all_episodes),num_episodes/statistics_separation_counter)
        wins_of_all_episodes = np.split(np.array(wins_of_all_episodes),num_episodes/statistics_separation_counter)
        statistics = np.array([rewards_all_episodes, valid_steps_all_episodes, invalid_steps_all_episodes, milestones_all_episodes, wins_of_all_episodes])


        # From here the logging starts
        gameInformation = env.getLoggingInformation()
        timeFormat = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) # thx to Metalshark: https://stackoverflow.com/questions/3220284/how-to-customize-the-time-format-for-python-logging

        simulationInformation = [timeFormat, action_space_size, state_space_size, q_table, num_episodes, max_steps_per_episode, learning_rate, discount_rate, 
                                exploration_rate, log_exploration_decay_rate, max_exploration_rate, log_min_exploration_rate, start_exploration_rate, log_notes,
                                statistics, statistics_separation_counter, total_steps, total_valid_steps, timeMeasurement]

        logger = Logger(gameInformation, simulationInformation)
        logger.createLog()

        self.q_table = q_table