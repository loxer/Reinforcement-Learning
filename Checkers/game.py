import numpy as np
import random


class Game:
    def __init__(self, size):
        self.size = size
        self.board = np.zeros([size, size], dtype=int)
        self.numberOfFields = size * size
        self.stonesPlayer1 = 0
        self.stonesPlayer2 = 0
        self.REWARD_VALID_STEP = 0
        self.REWARD_MILESTONE = 1
        self.REWARD_WON = 1
        self.REWARD_LOST = 0
        self.VERSION = 1
        self.prepareBoard()

    def reset(self):
        self.clearBoard()
        self.prepareBoard()
        state = 0
        for x in range(self.size):
            for y in range(self.size):                
                if self.board[x,y] == 1:
                    state += pow(2, x * self.size + y)
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
                if self.validField(x, y):       # check only the black fields
                    if self.board[x,y] == 1:
                        x = x // 2
                        y = y // 2
                        new_state += pow(2, x * self.size + y)  # new state is calculated from binary-to-decimal (board array is in decimal, states are in decimal)
        return new_state

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
        for y in range(self.size):
            if self.board[self.size-1, y] == 1:
                stonesFinished += 1
        
        return stonesFinished == self.stonesPlayer1


    def action_space(self):
        # ceiling divisions, thanks to Raymond Hettinger: https://stackoverflow.com/questions/33299093/how-to-perform-ceiling-division-in-integer-arithmetic
        stoneRowsPerPlayer = -((self.size - 2) // -2)
        self.stonesPerPlayer = -((stoneRowsPerPlayer * self.size) // -2)
        possibleActions = self.stonesPerPlayer * self.numberOfFields
        return possibleActions

    def state_space(self):
        possibleActions = pow(self.numberOfFields, 2)
        return possibleActions

    def action_space_sample(self):
        randomStone = random.randint(0, self.stonesPlayer1-1)
        randomField = random.randint(0, self.numberOfFields-1)
        return randomStone * self.numberOfFields + randomField
        
    def step(self, action):
        stone = action // self.numberOfFields           # which stone should be moved
        destination = action % self.numberOfFields
        destinationX = destination // self.size
        destinationY = destination % self.size
        # origin = -1

        new_state = self.getState()
        reward = 0
        done = False
        info = [False, False, False]       # [0] => bool, valid steps; [1] => bool, reached milestone; [2] => bool, game won
        
        for x in range(self.size):
            for y in range(self.size):
                
                # if self.board[x,y] == 1:
                #     new_state += pow(2, x * self.size + y)

                if self.board[x, y] == 1:
                    stone -= 1
                    if stone == -1:                     # find the stone on the board, which should be moved
                        moveIsValid = self.moveIsValid(1, x, y, destinationX, destinationY)

                        self.board[x,y] = 0
                        self.board[destinationX,destinationY] = 1
                        
                        # print("From: " + str(x) + "/" + str(y))
                        # print("To: " + str(destinationX) + "/" + str(destinationY))
                        # origin = self.arrayPosition(x, y)
                        
                        if moveIsValid:
                            reward = self.REWARD_VALID_STEP
                            info[0] = True                            

                            if destinationX == self.size - 1:      # piece is at the other side of the board => considered as 'milestone'
                                reward = self.REWARD_MILESTONE
                                done = self.isGameWon()
                                info[1] = True
                                # print("Milestone")

                                if done:                           # game has been won
                                    reward = self.REWARD_WON
                                    # print("Game won")
                                    info[2] = True
                                    break                                                                   

                        else:                                       # player lost, because of invalid move
                            reward = self.REWARD_LOST
                            done = True
                        break
        return new_state, reward, done, info

    def getBoard(self):
        return self.board

    def getLoggingInformation(self):
        """
        docstring
        """
        return [self.VERSION, self.numberOfFields, self.stonesPlayer1, self.REWARD_VALID_STEP, self.REWARD_MILESTONE, self.REWARD_WON, self.REWARD_LOST]