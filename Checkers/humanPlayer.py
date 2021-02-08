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

        new_line = "\n"
        indent = "         "        
        question = indent + "¿¿¿ "
        answer = indent + "=====>  "
        game_started = 2 * new_line + indent + "******* GAME STARTED *******" + 2 * new_line + self.print_board(game.getBoard(), indent)

        print(game_started)
        
        while user_keeps_playing:
            state = game.reset()

            while not(game_over):
                action = input(question + "What do you want to do?" + 2 * new_line)
                print("")

                if action == "new":
                    q_table = np.zeros((game.state_space(), game.action_space()))
                    print(answer + "New Q-Table created!" + 2 * new_line)

                elif action == "on":
                    if isinstance(q_table, int):
                        print(answer + self.q_table_missing() + 2 * new_line)
                    else:
                        advised_learning_enabled = True
                        print(answer + "All further moves will be used to improve the agent." + 2 * new_line)

                elif action == "off":
                    advised_learning_enabled = False
                    print(answer + "Agent will not learn from further moves." + 2 * new_line)
                
                elif action == "state":
                    print(answer + "Current state: " + str(game.getState()) + 2 * new_line)

                elif action == "tip":
                    print(answer + "Agent suggests: " + str(np.argmax(q_table[state,:])) + 2 * new_line)

                elif action == "max":
                    if isinstance(q_table, int):
                        print(answer + self.q_table_missing() + 2 * new_line)
                    else:
                        reward = np.argmax(q_table[state,:])
                        print(answer + "q_table[state,:]: " + str(q_table[state,:]) + 2 * new_line)
                        # print (indent + "Highest reward: " + str(np.argmax(q_table[,reward:])) + "\n")                    

                elif action == "print":
                    print_advised_learning_results = True
                    print(answer + "Reward and state will be printed now." + 2 * new_line)

                elif action == "print off":
                    print_advised_learning_results = False
                    print(answer + "Reward and state will NOT be printed." + 2 * new_line)

                elif action == "stop":
                    game_over = True
                    print(indent + "------ Match aborted ------" + 2 * new_line)

                elif action == "pass":
                    if isinstance(q_table, int):
                        print(answer + self.q_table_missing() + 2 * new_line)
                    else:
                        action = str(np.argmax(q_table[state,:]))
                        print(answer + "Agent's step: " + action + 2 * new_line)

                if action.isdigit():
                    action = int(action)

                    if action <= game.action_space():
                        new_state, reward, game_over, info = game.step(action)

                        print(answer + "New State: " + str(new_state) + new_line)

                        if advised_learning_enabled:    # Update Q-table
                            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

                        if print_advised_learning_results:
                            print(answer + "Reward: " + str(reward) + " || State: " + str(state) + new_line)

                        state = new_state       # Set new state
                        
                        print(self.print_board(game.getBoard(), indent))
                        print(indent + self.move_message(info))


            action = input(question + "Another round?" + indent + "(yes/no)" + new_line)
            if action == "yes":
                game_over = False                
                print(game_started)

            if action == "no":
                user_keeps_playing = False
                print(new_line + indent + "******* See you! *******" + 2 * new_line)                


    def print_board(self, board, indent):
        board_print = ""
        for x in range(len(board)):
            board_print += 2 * indent + str(board[x]) + "\n"
        return board_print


    def move_message(self, info):
        msg_for_user = ""

        if not(info[0]):
            msg_for_user += "--- Wrong move, game over! ---"
        elif info[2]:
            msg_for_user += "++++++ Awesome, you won the game! ++++++"
        elif info[1]:
            msg_for_user += "++++ Great, you made it to the other side! ++++"
        else:
            msg_for_user += "++ Your move was valid ++"

        return msg_for_user + "\n\n"


    def q_table_missing(self):
        return "No Q-Table found!"
