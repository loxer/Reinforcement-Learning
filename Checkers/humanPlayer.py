from game import *
import numpy as np


class HumanPlayer:
    def __init__(self, size, q_table):
        self.size = size


    def start(self):
        game = Game(self.size)
        state = game.reset()        
        game_over = False

        print(game.getBoard())

        while not(game_over):
            user_action = input('Make a move!\n')
            user_action = int(user_action)

            if user_action <= game.action_space():
                new_state, reward, game_over, info = game.step(user_action)
                print(game.getBoard())
                print("")

                if not(info[0]):
                    print("Wrong move, game over!")
                elif info[2]:
                    print("Awesome, you won the game!")
                elif info[1]:
                    print("Great, you made it to the other side!")
                else:
                    print("Your move was valid")





