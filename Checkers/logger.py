class Logger:
    def __init__(self, gameInformation, trainingInformation):
        self.gameInformation = gameInformation
        self.trainingInformation = trainingInformation
        self.success_rate_overall_valid_steps = 0
        self.overall_milestone_rate = 0
        self.overall_win_rate = 0
        self.statistic_string = ""
        self.FILE = ""


    def getData(self):
        return self.statistic_string, self.FILE


    def getHeadline(self, headline):
        return "\n\n" + "------------------------------------------- " + headline.upper() + " -------------------------------------------" + "\n"


    def getStatistics(self, statistics, statistics_separation_counter, num_episodes, total_steps, total_valid_steps, stonesPlayer, infoSeparator, newLine):

        rewards = statistics[0]
        valid_steps = statistics[1]
        invalid_steps = statistics[2]
        milestones = statistics[3]
        wins = statistics[4]

        self.success_rate_overall_valid_steps = total_valid_steps / total_steps

        counter = statistics_separation_counter
        logStatistics = ""
        
        # useage of format function found here: https://www.codespeedy.com/print-floats-to-a-specific-number-of-decimal-points-in-python/
        for episode_phase in range(num_episodes // statistics_separation_counter):
            sum_valid_steps = sum(valid_steps[episode_phase])
            sum_invalid_steps = sum(invalid_steps[episode_phase])
            sum_wins = int(sum(wins[episode_phase]))
            sum_milestones = sum(milestones[episode_phase])
            self.overall_win_rate += sum_wins
            self.overall_milestone_rate += sum_milestones

            sum_rewards = "{0:.2f}".format(sum(rewards[episode_phase] / statistics_separation_counter)) + infoSeparator
            percentage_valid_steps = "{0:.2f}".format(sum_valid_steps / (sum_valid_steps + sum_invalid_steps) * 100) + " %" + infoSeparator
            percentage_milestones = "{0:.2f}".format(sum_milestones / stonesPlayer / statistics_separation_counter * 100) + " %" + infoSeparator
            percentage_wins = "{0:.2f}".format(sum_wins / statistics_separation_counter * 100) + " %"
            
            # logStatistics += str(counter) + " | "
            logStatistics += "Rewards: " + sum_rewards
            logStatistics += "Valid Steps: " + percentage_valid_steps
            logStatistics += "Milestones: " + percentage_milestones
            logStatistics += "Wins: " + percentage_wins + " / " + str(sum_wins) + infoSeparator            
            logStatistics += "Episodes: " + str(counter - statistics_separation_counter) + " to " + str(counter)
            logStatistics += newLine
            counter += statistics_separation_counter

        self.overall_milestone_rate = self.overall_milestone_rate / stonesPlayer / num_episodes
        self.overall_win_rate = self.overall_win_rate / num_episodes

        valid_steps_total = "Valid Steps Rate: " + "{0:.2f}".format(self.success_rate_overall_valid_steps * 100) + " %"
        milestones_total = "Milestone Rate: " + "{0:.2f}".format(self.overall_milestone_rate * 100) + " %"
        wins_total = "Winning Rate: " + "{0:.2f}".format(self.overall_win_rate * 100) + " %"

        logStatistics += newLine + "Total " + valid_steps_total
        logStatistics += newLine + "Total " + milestones_total
        logStatistics += newLine + "Total " + wins_total

        self.statistic_string = infoSeparator + valid_steps_total + infoSeparator + milestones_total + infoSeparator + wins_total + infoSeparator
        return logStatistics


    def writeToFile(self, version, startingTime, size, logMessage):        
        valid_steps_rate = "__" + "{0:.2f}".format(self.success_rate_overall_valid_steps * 100)#.replace('.' , '-')
        win_rate = "__" + "{0:.2f}".format(self.overall_win_rate * 100)#.replace('.' , '-')
        file_format = ".txt"
        FILE = "Checkers\\Logs\\Version_" + str(version) + "\\" + str(size) + "\\" + startingTime + valid_steps_rate + win_rate + file_format
        logFile = open(FILE,"w+")
        logFile.write(logMessage)
        self.FILE = FILE
        print("\nFile can be found at: " + FILE)


    def createLog(self):        
        version = self.gameInformation[0]
        numberOfFields = self.gameInformation[1]
        stonesPlayer1 = self.gameInformation[2]
        rewardValidStep = self.gameInformation[3]
        rewardMilestone = self.gameInformation[4]
        rewardWon = self.gameInformation[5]
        rewardLost = self.gameInformation[6]

        startingTime = self.trainingInformation[0]
        action_space_size = self.trainingInformation[1]
        state_space_size = self.trainingInformation[2]
        # q_table = self.trainingInformation[3]
        num_episodes = self.trainingInformation[4]
        # max_steps_per_episode = self.trainingInformation[5]
        learning_rate = self.trainingInformation[6]
        discount_rate = self.trainingInformation[7]
        # exploration_rate = self.trainingInformation[8]
        exploration_decay_rate = self.trainingInformation[9]
        max_exploration_rate = self.trainingInformation[10]
        min_exploration_rate = self.trainingInformation[11]
        start_exploration_rate = self.trainingInformation[12]
        notes = self.trainingInformation[13]
        statistics = self.trainingInformation[14]
        statistics_separation_counter = self.trainingInformation[15]
        total_steps = self.trainingInformation[16]
        total_valid_steps = self.trainingInformation[17]
        timeMeasurement = self.trainingInformation[18]
        size = self.trainingInformation[19]
        current_training_episode = self.trainingInformation[20]
        num_trainings = self.trainingInformation[21]
        endingTime = self.trainingInformation[22]
        create_log_file = self.trainingInformation[23]

        newLine = "\n"
        infoSeparator = " || "

        logMessage = "******************************************** " + "REINFORCEMENT LEARNING AI LOGBOOK" + " ********************************************" + newLine
        logMessage += startingTime + infoSeparator + "Version: " + str(version) + infoSeparator +  "Training Episode: " + str(current_training_episode) + "/" + str(num_trainings) + newLine


        if notes:
            logMessage += self.getHeadline("NOTES")
            logMessage += notes    


        logMessage += self.getHeadline("GAME INFORMATION")
        logMessage += "Number of Fields: " + str(numberOfFields) + infoSeparator + "Number of Pieces Per Player: " + str(stonesPlayer1) + newLine
        
        logMessage += newLine +	"        REWARDS for..." + newLine + "...Valid Steps: " + str(rewardValidStep) + infoSeparator + "...Milestones: " + str(rewardMilestone)
        logMessage += infoSeparator + "...Winning: " + str(rewardWon) + infoSeparator + "...Loosing: " + str(rewardLost) + newLine


        logMessage += self.getHeadline("TRAINING INFORMATION")
        logMessage += "Action Space Size: " + str(action_space_size) + infoSeparator + "State Space Size: " + str(state_space_size) + infoSeparator + "Q-Table Size: " + str(action_space_size * state_space_size) + newLine        
        logMessage += "Learning Rate: " + str(learning_rate) + infoSeparator + "Discount Rate: " + str(discount_rate) + newLine
        logMessage += "Number of Episodes: " + str(num_episodes) + newLine
        # logMessage += infoSeparator + "Maximum Steps per Episode: " + str(max_steps_per_episode) + newLine

        logMessage += newLine +	"        EXPLORATION..." + newLine + "...Starting Rate: " + str(start_exploration_rate) + infoSeparator + "...Maximum Rate: " + str(max_exploration_rate)
        logMessage += infoSeparator + "...Minimum Rate: " + str(min_exploration_rate) + infoSeparator + "...Decay Rate: " + str(exploration_decay_rate) + newLine        


        logMessage += self.getHeadline("STATISTICS PER " + str(statistics_separation_counter) + " EPISODES")
        logMessage += self.getStatistics(statistics, statistics_separation_counter, num_episodes, total_steps, total_valid_steps, stonesPlayer1, infoSeparator, newLine)
        logMessage += 2*newLine + "Time for Processing " + str(num_episodes) + " Episodes: " + str(timeMeasurement) + " seconds"
        logMessage += newLine + "Average Time for 1000 Episodes: " + str(timeMeasurement / num_episodes * 1000) + " seconds"

        logMessage += self.getHeadline("LOG END")
    
        print(logMessage)
        print("Training ended at: " + endingTime)

        if create_log_file:
            self.writeToFile(version, startingTime, size, logMessage)
        
        print(2*newLine)