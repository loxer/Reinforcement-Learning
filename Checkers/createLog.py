class CreateLog:
    def __init__(self, gameInformation, simulationInformation):
        self.gameInformation = gameInformation
        self.simulationInformation = simulationInformation


    def getHeadline(self, headline):
        return "\n\n" + "--------------------------- " + headline.upper() + " ---------------------------" + "\n"


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
        exploration_rate = self.simulationInformation[8]
        exploration_decay_rate = self.simulationInformation[9]
        max_exploration_rate = self.simulationInformation[10]
        min_exploration_rate = self.simulationInformation[11]
        start_exploration_rate = self.simulationInformation[12]

        logMessage = timeFormat + infoSeparator + "Version: " + str(version) + newLine
        logMessage += self.getHeadline("GAME INFORMATION")
        logMessage += "Number of Fields: " + str(numberOfFields) + infoSeparator + "Number of Pieces of Player: " + str(stonesPlayer1) + newLine
        logMessage += newLine +	"        REWARDS for..." + newLine + "...Valid Steps: " + str(rewardValidStep) + infoSeparator + "...Milestones: " + str(rewardMilestone)
        logMessage += infoSeparator + "...Winning: " + str(rewardWon) + infoSeparator + "...Loosing: " + str(rewardLost) + newLine
        logMessage += self.getHeadline("SIMULATION INFORMATION")
        logMessage += "Action Space Size: " + str(action_space_size) + infoSeparator + "State Space Size: " + str(state_space_size) + infoSeparator + "Q-Table Size: " + str(action_space_size * state_space_size) + newLine
        logMessage += "Number of Episodes: " + str(num_episodes) + newLine
        # logMessage += infoSeparator + "Maximum Steps per Episode: " + str(max_steps_per_episode) + newLine
        logMessage += "Learning Rate: " + str(learning_rate) + infoSeparator + "Discount Rate: " + str(discount_rate) + newLine
        logMessage += newLine +	"        EXPLORATION..." + newLine + "Starting Rate: " + str(start_exploration_rate) + infoSeparator + "Decay Rate: " + str(exploration_decay_rate)
        logMessage += infoSeparator + "...Maximum Rate: " + str(max_exploration_rate) + infoSeparator + "...Minimum Rate: " + str(min_exploration_rate) + newLine


        return logMessage

