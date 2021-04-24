from logger import *
import numpy as np
import random
import time
import timeit

class Trainer:
    def __init__(self, board, training_settings, logging_settings):
        self.board = board
        self.training_settings = training_settings
        self.logging_settings = logging_settings
        self.learning_rate = training_settings[4]
        self.discount_rate = training_settings[5]
        self.algorithm = self.training_settings[10]
        self.step_memory = []


    def run(self, train_new_agents):
        board_size = self.training_settings[0]
        num_trainings = self.training_settings[1]
        
        
        num_episodes = self.training_settings[2]
        max_steps_per_episode = self.training_settings[3]

        agents = []
        agents_data = []


        for training in range(num_trainings):
            if train_new_agents:
                self.create_new_agent()

            start_exploration_rate = exploration_rate = self.training_settings[6]
            max_exploration_rate = self.training_settings[7]
            min_exploration_rate = float(self.training_settings[8])
            exploration_decay_rate = float(self.training_settings[9])

            rewards_all_episodes = []
            invalid_steps_all_episodes = []
            valid_steps_all_episodes = []
            milestones_all_episodes = []
            wins_of_all_episodes = []
        
            self.print_training_starter(training, num_trainings)
            startingTime = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()) # thx to Metalshark: https://stackoverflow.com/questions/3220284/how-to-customize-the-time-format-for-python-logging
            print("\n\nTraining started at: " + startingTime + "\n\n")
            timeMeasurement = timeit.default_timer()


            # <------------ MOST PART STARTING FROM HERE IS TAKEN FROM: https://deeplizard.com/learn/video/HGeI30uATws ------------>

            # Q-learning algorithm
            for episode in range(num_episodes):
                state = self.board.reset()
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
                        action = np.argmax(self.q_table[state,:])
                    else:
                        action = self.board.action_space_sample()
                        
                    # Take new action
                    new_state, reward, done, info = self.board.step(action)

                    # Update Q-table
                    if self.algorithm[0]:
                        self.q_learning_algorithm(state, new_state, action, reward)
                    if self.algorithm[1]:
                        self.save_step(state, new_state, action, reward)
                                
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
                    
                    # Set new state
                    state = new_state

                # Exploration rate decay
                exploration_rate = min_exploration_rate + \
                    (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
                
                # Add current episode reward to total rewards list
                rewards_all_episodes.append(rewards_current_episode)
                valid_steps_all_episodes.append(valid_steps_current_episode)
                invalid_steps_all_episodes.append(invalid_steps_current_episodes)
                milestones_all_episodes.append(milestones_current_episodes)

                # <----------------------------------------------- UNTIL HERE ----------------------------------------------->

                if self.algorithm[1]:
                    self.advanced_algorithm()


            # Preparing statistics for log
            timeMeasurement = timeit.default_timer() - timeMeasurement
            total_valid_steps = sum(valid_steps_all_episodes)
            total_steps = total_valid_steps + sum(invalid_steps_all_episodes)

            statistics_separation_counter = self.logging_settings[1]
            rewards_all_episodes = np.split(np.array(rewards_all_episodes),num_episodes/statistics_separation_counter)
            valid_steps_all_episodes = np.split(np.array(valid_steps_all_episodes),num_episodes/statistics_separation_counter)
            invalid_steps_all_episodes = np.split(np.array(invalid_steps_all_episodes),num_episodes/statistics_separation_counter)
            milestones_all_episodes = np.split(np.array(milestones_all_episodes),num_episodes/statistics_separation_counter)
            wins_of_all_episodes = np.split(np.array(wins_of_all_episodes),num_episodes/statistics_separation_counter)
            statistics = np.array([rewards_all_episodes, valid_steps_all_episodes, invalid_steps_all_episodes, milestones_all_episodes, wins_of_all_episodes])


            # From here the logging starts
            gameInformation = self.board.getLoggingInformation()        
            endingTime = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())        

            trainingInformation = [startingTime, self.board.action_space(), self.board.state_space(), self.q_table, num_episodes, max_steps_per_episode, self.learning_rate, self.discount_rate, 
                                    exploration_rate, self.training_settings[9], max_exploration_rate, self.training_settings[8], start_exploration_rate, self.logging_settings[0],
                                    statistics, statistics_separation_counter, total_steps, total_valid_steps, timeMeasurement, board_size, training, str(num_trainings), endingTime, self.logging_settings[3]]
            
            self.logger = Logger(gameInformation, trainingInformation)
            self.logger.createLog()
            agents_data.append(self.logger.getData())
            agents.append(self.q_table)

        return agents, agents_data
        

    def create_new_agent(self):
        self.q_table = np.zeros((self.board.state_space(), self.board.action_space()), dtype=np.float32)


    def set_agent(self, q_table):
        self.q_table = q_table


    def get_agent(self):
        return self.q_table


    def save_step(self, state, new_state, action, reward):
        self.step_memory.append([state, new_state, action, reward])


    def reset_step_memory(self):
        self.step_memory.clear()


    def q_learning_algorithm(self, state, new_state, action, reward):
        self.q_table[state, action] = self.q_table[state, action] * (1 - self.learning_rate) + \
                        self.learning_rate * (reward + self.discount_rate * np.max(self.q_table[new_state, :]))


    def advanced_algorithm(self):
        for step in range(len(self.step_memory)-1,-1,-1):      # just iterate in reverse order through the list
            state = self.step_memory[step][0]
            new_state = self.step_memory[step][1]
            action = self.step_memory[step][2]
            reward = self.step_memory[step][3]
            self.q_learning_algorithm(state, new_state, action, reward)
        
        self.reset_step_memory()


    def advise_agent(self, state, new_state, action, reward, game_over):
        if self.algorithm[0]:
            self.q_learning_algorithm(state, new_state, action, reward)
        if self.algorithm[1]:
            self.save_step(state, new_state, action, reward)
            if game_over:
                self.advanced_algorithm()



    def print_training_starter(self, training, num_trainings):
        print("\n")
        print("                  ******************************************\n")
        print("                  ****** TRAINING EPISODE " + str(training+1) + "/" + str(num_trainings) + " STARTED ******\n")
        print("                  ******************************************\n\n")