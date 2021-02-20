import numpy as np
import random


class Game:
    def __init__(self, size, reward_settings):
        self.size = size
        self.board = np.zeros([size, size], dtype=int)
        self.numberOfFields = size * size
        self.stoneRowsPerPlayer = -((self.size - 2) // -2) # ceiling divisions, thanks to Raymond Hettinger: https://stackoverflow.com/questions/33299093/how-to-perform-ceiling-division-in-integer-arithmetic
        self.stonesPerPlayer = 0
        self.stonesPlayer1 = 0
        self.stonesPlayer2 = 0
        self.reward_valid_step = reward_settings[0]
        self.reward_milestone = reward_settings[1]
        self.reward_won = reward_settings[2]
        self.reward_lost = reward_settings[3]
        self.VERSION = 2
        self.prepareBoard()


    def reset(self):
        self.clearBoard()
        self.prepareBoard()
        return self.getState()


    def prepareBoard(self):
        for x in range(len(self.board)):
            for y in range(len(self.board[x])):
                if self.validField(x, y):
                    if x < self.size / 2 - 1:   # Rows of Player 1
                        self.board[x,y] = 1
                        self.stonesPlayer1 += 1
                        # print(str(x) + "/" + str(y))
                    # if x > self.size / 2:     # Rows of Player 2
                    #     self.board[x,y] = 3
                    #     self.stonesPlayer1 += 1
        self.stonesPerPlayer = self.stonesPlayer1
        # print(self.board)


    def clearBoard(self):
        self.stonesPlayer1 = 0
        self.stonesPlayer2 = 0
        for x in range(len(self.board)):            
            for y in range(len(self.board[x])):
                self.board[x,y] = 0


    def validField(self, x, y):
        return (x + y) % 2 == 0         # every black field (defined as an even number here) is a valid field


    def getState(self):
        new_state = 0
        for x in range(self.size):
            for y in range(self.size):
                    if self.board[x,y] == 1:                        
                        y_calc = y // 2                                   
                        new_state += pow(2, x * self.size // 2 + y_calc)  # new state is calculated from binary-to-decimal (board array is in decimal, states are in decimal)
                        # print(str(x) + "/" + str(y) + " | " + str(x) + "/" + str(y_calc) + " | Exponent: " + str(x * self.size // 2 + y_calc) + " | Current sum: " + str(new_state))
                        y += 1      # skip the white fields
        return new_state - 1


    def arrayPosition(self, x, y):
        return x * len(self.board) + y


    def moveIsValid(self, player, stoneX, stoneY, toFieldX, toFieldY):
        if player == 1:
            if stoneX + 1 == toFieldX and toFieldX < self.size and (stoneY + 1 == toFieldY or stoneY - 1 == toFieldY) and toFieldY < self.size and toFieldY >= 0 and self.board[toFieldX,toFieldY] == 0:
                return True
            else:   
                return False  


    def isGameWon(self):
        stonesFinished = 0
        for x in range(self.stoneRowsPerPlayer):
            for y in range(self.size):
                if self.board[self.size-x-1, y] == 1:
                    stonesFinished += 1
        
        return stonesFinished == self.stonesPerPlayer


    def action_space(self):        
        return self.stonesPerPlayer * self.numberOfFields   # highest number of possible actions


    def state_space(self):
        # ceiling divisions, thanks to Raymond Hettinger: https://stackoverflow.com/questions/33299093/how-to-perform-ceiling-division-in-integer-arithmetic
        valid_fields = -(self.numberOfFields // -2)
        highest_possible_state = 0
        
        for i in range(self.stonesPerPlayer):            
            highest_possible_state += pow(2, valid_fields - i - 1)

        return highest_possible_state


    def action_space_sample(self):
        randomStone = random.randint(0, self.stonesPlayer1-1)
        randomField = random.randint(0, self.numberOfFields-1)
        return randomStone * self.numberOfFields + randomField
        

    def step(self, action, move = True):
        stone = action // self.numberOfFields           # which stone should be moved
        destination = action % self.numberOfFields
        destinationX = destination // self.size
        destinationY = destination % self.size

        new_state = 0
        reward = 0
        done = False
        info = [False, False, False]       # [0] => bool, valid steps; [1] => bool, reached milestone; [2] => bool, game won
        
        for x in range(self.size):
            for y in range(self.size):

                if self.board[x, y] == 1:
                    stone -= 1

                    if stone == -1:                     # find the stone on the board, which should be moved
                        moveIsValid = self.moveIsValid(1, x, y, destinationX, destinationY)

                        if not(move):                   # check for testing
                            info[0] = moveIsValid
                        else:
                            self.board[x,y] = 0
                            self.board[destinationX,destinationY] = 1

                            if moveIsValid:
                                new_state = self.getState()
                                reward = self.reward_valid_step
                                info[0] = True                            

                                if destinationX == self.size - self.stoneRowsPerPlayer: # piece is at the other side of the board => considered as 'milestone'
                                    reward = self.reward_milestone
                                    done = self.isGameWon()
                                    info[1] = True
                                    # print("Milestone")

                                    if done:                           # game has been won
                                        reward = self.reward_won
                                        # print("Game won")
                                        info[2] = True
                                        break                                                                   

                            else:                                       # player lost, because of invalid move
                                reward = self.reward_lost
                                done = True
                            break

        return new_state, reward, done, info


    def getBoard(self):
        return self.board


    def getLoggingInformation(self):
        """
        docstring
        """
        return [self.VERSION, self.numberOfFields, self.stonesPlayer1, self.reward_valid_step, self.reward_milestone, self.reward_won, self.reward_lost]