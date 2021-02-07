from game import *
import numpy as np


class HumanPlayer:
    def __init__(self, size):
        self.size = size


    def start(self, q_table, learning_attributes):
        game = Game(self.size)
        user_keeps_playing = True
        game_over = False
        advised_learning_enabled = False
        print_advised_learning_results = False
        learning_rate = learning_attributes[0]
        discount_rate = learning_attributes[1]
        indent = "         --> "

        print("\n\n *** GAME STARTED ***\n")
        print(game.getBoard())
        print("")

        while user_keeps_playing:
            state = game.reset()

            while not(game_over):
                action = input(indent + "What do you want to do?\n")
                print("")

                if action == "new":
                    q_table = np.zeros((game.state_space(), game.action_space()))                    
                    print(indent + "New Q-Table created!\n")

                elif action == "on":
                    advised_learning_enabled = True
                    print(indent + "All further moves will be used to improve the agent.\n")

                elif action == "off":
                    advised_learning_enabled = False
                    print(indent + "Agent will not learn from further moves.\n")
                
                elif action == "state":
                    print(indent + "Current state: " + str(state) + "\n")

                elif action == "tip":
                    print(indent + "Agent suggests: " + str(np.argmax(q_table[state,:])) + "\n")

                # elif action == "max":
                #     reward = np.argmax(q_table[state,:])
                #     print (indent + "Highest reward: " + str(np.argmax(q_table[,reward:])) + "\n")

                elif action == "print":
                    print_advised_learning_results = True
                    print(indent + "Reward and state will be printed now.\n")

                elif action == "print off":
                    print_advised_learning_results = False
                    print(indent + "Reward and state will NOT be printed.\n")

                elif action == "stop":
                    game_over = True
                    print("--- Match aborted ---\n")

                elif action == "pass":
                    action = np.argmax(q_table[state,:])
                    print(indent + "Agent's step: " + str(action) + "\n")

                if action.isdigit():
                    action = int(action)

                    if action <= game.action_space():
                        new_state, reward, game_over, info = game.step(action)

                        if advised_learning_enabled:    # Update Q-table
                            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

                        if print_advised_learning_results:
                            print(indent + "Reward: " + str(reward) + " || State: " + str(state) + "\n")                            

                        state = new_state       # Set new state
                        
                        print(game.getBoard())
                        self.move_message(info)
                        
                        

            action = input(indent + "Another round?\n")
            if action == "yes":
                game_over = False

            if action == "no":
                user_keeps_playing = False

            print("")


    def move_message(self, info):
        msg_for_user = "\n\n"

        if not(info[0]):
            msg_for_user += "Wrong move, game over!"
        elif info[2]:
            msg_for_user += "Awesome, you won the game!"
        elif info[1]:
            msg_for_user += "Great, you made it to the other side!"
        else:
            msg_for_user += "Your move was valid"

        msg_for_user += "\n"


