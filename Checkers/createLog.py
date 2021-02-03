class CreateLog:
    def __init__(self, gameInformation, simulationInformation):
        self.gameInformation = gameInformation
        self.simulationInformation = simulationInformation
        self.success_rate_overall_valid_steps = 0
        self.overall_milestone_rate = 0
        self.overall_win_rate = 0


    def getHeadline(self, headline):
        return "\n\n" + "------------------------------------------- " + headline.upper() + " -------------------------------------------" + "\n"


    def getStatistics(self, statistics, statistics_separation_counter, num_episodes, total_steps, total_valid_steps, infoSeparator, newLine):

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
            sum_wins = sum(wins[episode_phase])
            sum_milestones = sum(milestones[episode_phase])
            self.overall_win_rate += sum_wins
            self.overall_milestone_rate += sum_milestones

            sum_rewards = "{0:.5f}".format(sum(rewards[episode_phase] / statistics_separation_counter)) + infoSeparator
            percentage_valid_steps = "{0:.2f}".format(sum_valid_steps / (sum_valid_steps + sum_invalid_steps) * 100) + " %" + infoSeparator
            percentage_milestones = "{0:.2f}".format(sum_milestones / statistics_separation_counter * 100) + " %" + infoSeparator
            percentage_wins = "{0:.3f}".format(sum_wins / statistics_separation_counter * 100) + " %"
            

            # logStatistics += str(counter) + " | "
            logStatistics += "Rewards: " + sum_rewards
            logStatistics += "Valid Steps: " + percentage_valid_steps
            logStatistics += "Milestones: " + percentage_milestones
            logStatistics += "Wins: " + percentage_wins + " / " + str(sum(wins[episode_phase])) + infoSeparator
            logStatistics += "Episodes: " + str(counter - statistics_separation_counter) + " to " + str(counter)
            logStatistics += newLine
            counter += statistics_separation_counter

        self.overall_milestone_rate = self.overall_milestone_rate / num_episodes
        self.overall_win_rate = self.overall_win_rate / num_episodes

        logStatistics += newLine + "Valid Steps in Total: " + "{0:.2f}".format(self.success_rate_overall_valid_steps * 100) + " %"
        logStatistics += newLine + "Milestone Rate in Total: " + "{0:.2f}".format(self.overall_milestone_rate * 100) + " %"
        logStatistics += newLine + "Winning Rate in Total: " + "{0:.3f}".format(self.overall_win_rate * 100) + " %"

        return logStatistics


    def writeToFile(self, version, timeFormat, logMessage):        
        valid_steps_rate = "__" + "{0:.2f}".format(self.success_rate_overall_valid_steps * 100)#.replace('.' , '-')
        win_rate = "__" + "{0:.3f}".format(self.overall_win_rate * 100)#.replace('.' , '-')
        file_format = ".txt"
        FILE = "Logs\\Version_" + str(version) + "\\" + timeFormat + valid_steps_rate + win_rate + file_format
        logFile = open(FILE,"w+")
        logFile.write(logMessage)


    def getLog(self):        
        version = self.gameInformation[0]
        numberOfFields = self.gameInformation[1]
        stonesPlayer1 = self.gameInformation[2]
        rewardValidStep = self.gameInformation[3]
        rewardMilestone = self.gameInformation[4]
        rewardWon = self.gameInformation[5]
        rewardLost = self.gameInformation[6]

        timeFormat = self.simulationInformation[0]
        action_space_size = self.simulationInformation[1]
        state_space_size = self.simulationInformation[2]
        # q_table = self.simulationInformation[3]
        num_episodes = self.simulationInformation[4]
        # max_steps_per_episode = self.simulationInformation[5]
        learning_rate = self.simulationInformation[6]
        discount_rate = self.simulationInformation[7]
        # exploration_rate = self.simulationInformation[8]
        exploration_decay_rate = self.simulationInformation[9]
        max_exploration_rate = self.simulationInformation[10]
        min_exploration_rate = self.simulationInformation[11]
        start_exploration_rate = self.simulationInformation[12]
        notes = self.simulationInformation[13]
        statistics = self.simulationInformation[14]
        statistics_separation_counter = self.simulationInformation[15]
        total_steps = self.simulationInformation[16]
        total_valid_steps = self.simulationInformation[17]
        timeMeasurement = self.simulationInformation[18]

        newLine = "\n"
        infoSeparator = " || "

        logMessage = "******************************************** " + "REINFORCEMENT LEARNING AI LOGBOOK" + " ********************************************" + newLine
        logMessage += timeFormat + infoSeparator + "Version: " + str(version) + newLine


        if notes:
            logMessage += self.getHeadline("NOTES")
            logMessage += notes    


        logMessage += self.getHeadline("GAME INFORMATION")
        logMessage += "Number of Fields: " + str(numberOfFields) + infoSeparator + "Number of Pieces of Player: " + str(stonesPlayer1) + newLine
        
        logMessage += newLine +	"        REWARDS for..." + newLine + "...Valid Steps: " + str(rewardValidStep) + infoSeparator + "...Milestones: " + str(rewardMilestone)
        logMessage += infoSeparator + "...Winning: " + str(rewardWon) + infoSeparator + "...Loosing: " + str(rewardLost) + newLine


        logMessage += self.getHeadline("SIMULATION INFORMATION")
        logMessage += "Action Space Size: " + str(action_space_size) + infoSeparator + "State Space Size: " + str(state_space_size) + infoSeparator + "Q-Table Size: " + str(action_space_size * state_space_size) + newLine        
        logMessage += "Learning Rate: " + str(learning_rate) + infoSeparator + "Discount Rate: " + str(discount_rate) + newLine
        logMessage += "Number of Episodes: " + str(num_episodes) + newLine
        # logMessage += infoSeparator + "Maximum Steps per Episode: " + str(max_steps_per_episode) + newLine

        logMessage += newLine +	"        EXPLORATION..." + newLine + "...Starting Rate: " + str(start_exploration_rate) + infoSeparator + "...Maximum Rate: " + str(max_exploration_rate)
        logMessage += infoSeparator + "...Minimum Rate: " + str(min_exploration_rate) + infoSeparator + "...Decay Rate: " + str(exploration_decay_rate) + newLine        


        logMessage += self.getHeadline("STATISTICS PER " + str(statistics_separation_counter) + " EPISODES")
        logMessage += self.getStatistics(statistics, statistics_separation_counter, num_episodes, total_steps, total_valid_steps, infoSeparator, newLine)
        logMessage += 2*newLine + "Time for Processing " + str(num_episodes) + " Episodes: " + str(timeMeasurement) + " seconds"
        logMessage += newLine + "Average Time for 1000 Episodes: " + str(timeMeasurement / num_episodes * 1000) + " seconds"

        logMessage += self.getHeadline("LOG END")
    

        self.writeToFile(version, timeFormat, logMessage)
        print(logMessage)