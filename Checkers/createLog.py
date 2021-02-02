class CreateLog:
    def __init__(self, gameInformation, simulationInformation):
        self.gameInformation = gameInformation
        self.simulationInformation = simulationInformation


    def getHeadline(self, headline):
        return "\n\n" + "------------------------------------------- " + headline.upper() + " -------------------------------------------" + "\n"


    def getStatistics(self, statistics, statistics_separation_counter, num_episodes, infoSeparator, newLine):

        rewards = statistics[0]
        valid_steps = statistics[1]
        invalid_steps = statistics[2]
        milestones = statistics[3]
        wins = statistics[4]

        counter = statistics_separation_counter
        logStatistics = ""
        
        for episode_phase in range(num_episodes // statistics_separation_counter):
            percentage_valid_steps = " / " + ("{0:.1f}".format(sum(valid_steps[episode_phase]) / sum(invalid_steps[episode_phase]) * 100)) + " %" + infoSeparator


            logStatistics += str(counter) + " | "
            logStatistics += "Valid Steps: " + str(sum(valid_steps[episode_phase])) + percentage_valid_steps
            logStatistics += "Milestones: " + str(sum(milestones[episode_phase])) + " / " + str(sum(milestones[episode_phase]/statistics_separation_counter) * 100) + " %" + infoSeparator
            logStatistics += "Wins: " + str(sum(wins[episode_phase])) + " / " + str(sum(wins[episode_phase]/statistics_separation_counter) * 100) + " %" + infoSeparator
            logStatistics += "Rewards: " + str(sum(rewards[episode_phase]/statistics_separation_counter))
            logStatistics += " | " + str(counter)
            logStatistics += newLine
            counter += statistics_separation_counter

        return logStatistics


    def writeToFile(self, version, timeFormat, logMessage):
        FILE = "Logs\\Version_" + str(version) + "\\" + timeFormat + "_123.txt"
        logFile = open(FILE,"w+")
        logFile.write(logMessage)


    def getLog(self):        
        newLine = "\n"
        infoSeparator = " || "

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

        logMessage = "******************************************** " + "REINFORCEMENT LEARNING AI LOGBOOK" + " ********************************************" + "\n"
        logMessage += timeFormat + infoSeparator + "Version: " + str(version) + newLine

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
        logMessage += self.getStatistics(statistics, statistics_separation_counter, num_episodes, infoSeparator, newLine)


        if notes:
            logMessage += self.getHeadline("EXTRA NOTES")
            logMessage += notes        
        

        self.writeToFile(version, timeFormat, logMessage)
        print(logMessage)

        

