from simulation import *
import numpy as np
import h5py


class HumanPlayer:
    def __init__(self, size):
        self.size = size


    def start(self, board, simulation_settings, logging_settings, agent_save_path):
        game = board
        q_table = np.zeros((game.state_space(), game.action_space()))
        
        board_size = simulation_settings[0]        
        learning_rate = simulation_settings[4]
        discount_rate = simulation_settings[5]

        programming_running = True
        user_keeps_playing = False
        game_over = False
        question_answered = False
        advised_learning_enabled = False
        print_advised_learning_results = False
        

        new_line = "\n"
        indent = "         "        
        question = indent + "¿¿¿ "
        answer = indent + "=====>  "
        game_started = 2 * new_line + indent + "******* GAME STARTED *******" + 2 * new_line + self.print_board(game.getBoard(), indent)

        print(game_started)
        while programming_running:
            action = input(question + "What do you want to do?" + 2 * new_line)
            print("")

            if action == "play":
                user_keeps_playing = True
                game_over = False

            elif action == "load":
                with h5py.File(agent_save_path, "r") as hdf:
                    agent_list = list(hdf.keys())
                    print("Available agents: ")
                    print(agent_list)
                    loaded_agent = hdf.get("agent1")
                    q_table = np.array(loaded_agent)

            elif action == "train":
                agents = []
                agents_data = []
                num_episodes = simulation_settings[1]
                for simulation_episode in range(num_episodes):
                    simulation = Simulation()
                    simulation.run(board, simulation_settings, logging_settings, str(simulation_episode + 1))
                    agents.append(simulation.getAgent())
                    agents_data.append(simulation.get_logging_data())
                
                while not(question_answered):
                    print(new_line + question + "Do you want to keep any of the following agents?" + new_line)
                    for i in range(num_episodes):
                        print(answer + str(i+1) + ": " + agents_data[i][0] + "File: " + agents_data[i][1])
                    action = input()
                    
                    if action == "no":
                        question_answered = True
                    elif action == "save":
                        with h5py.File(agent_save_path, "w") as hdf:
                            hdf.create_dataset("agent1", data = agents[0])

                    elif action.isdigit():
                        action = int(action)
                        if action > 0 and action <= num_episodes:
                            q_table = agents[action-1]
                            question_answered = True
                        else:
                            print(answer + "This agent does NOT exist!" + new_line)
                print(new_line)

            elif action == "close":
                programming_running = False


            while user_keeps_playing:
                state = game.reset()

                while not(game_over):
                    action = input(question + "What do you want to do?" + 2 * new_line)
                    print("")                

                    if action == "new":
                        q_table = np.zeros((game.state_space(), game.action_space()))
                        print(answer + "New Q-Table created!" + 2 * new_line)

                    elif action == "on":
                            advised_learning_enabled = True
                            print(answer + "All further moves will be used to improve the agent." + 2 * new_line)

                    elif action == "off":
                        advised_learning_enabled = False
                        print(answer + "Agent will not learn from further moves." + 2 * new_line)
                    
                    elif action == "state":
                        print(answer + "Current state: " + str(game.getState()) + 2 * new_line)

                    elif action == "tip":
                        pos_values, _ = self.get_highest_values(q_table, state)
                        print(answer + "Agent suggests: " + pos_values + 2 * new_line)

                    elif action == "table":
                            reward = np.argmax(q_table[state,:])
                            print(self.print_q_table(q_table, state, indent, answer, new_line))               

                    elif action == "print":
                        print_advised_learning_results = True
                        print(answer + "Reward and state will be printed now." + 2 * new_line)

                    elif "check" in action:
                        action = [int(word) for word in action.split() if word.isdigit()]    # Thx to Srikar Appalaraju: https://stackoverflow.com/questions/16009861/get-digits-from-string
                        if len(action) == 0:
                            action = answer + "No digits found." + 2 * new_line
                        else:
                            action = action[0]
                            if action <= game.action_space():
                                _, _, _, info = game.step(action, False)
                                if info[0] == False:
                                    action = indent + "--- This move would be INVALED ---" + 2 * new_line
                                else:
                                    action = indent + "+++ This move will be FINE +++" + 2 * new_line
                            else:
                                action = indent + "--- This move would be OUT OF RANGE ---" + 2 * new_line
                        print(action)

                    elif action == "hint":
                        for i in range(board_size + 1, game.action_space()):                        
                            _, _, _, info = game.step(i, False)
                            # print(str(i) + ": " + str(info[0]))
                            if info[0] == True:
                                print(answer + "A valid move will be: " + str(i) + 2 * new_line)
                                break

                    elif action == "print off":
                        print_advised_learning_results = False
                        print(answer + "Reward and state will NOT be printed." + 2 * new_line)

                    elif action == "board":
                        print(self.print_board(game.getBoard(), indent))

                    elif action == "stop":
                        game_over = True
                        print(indent + "------ Match aborted ------" + 2 * new_line)

                    elif action == "pass":
                            action = str(np.argmax(q_table[state,:]))
                            print(answer + "Agent's step: " + action + 2 * new_line)

                    if action.isdigit():
                        action = int(action)

                        if action <= game.action_space():
                            new_state, reward, game_over, info = game.step(action)

                            if advised_learning_enabled:    # Update Q-table
                                q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

                            if print_advised_learning_results:
                                print(answer + "Reward: " + str(reward) + " || State: " + str(state) + new_line)

                            state = new_state       # Set new state
                            
                            print(self.print_board(game.getBoard(), indent))
                            print(indent + self.move_message(info))


                action = input(new_line + question + "Another round?" + indent + "(yes/no)" + new_line)
                if action == "yes":
                    game_over = False
                    print(game_started)

                elif action == "no":
                    user_keeps_playing = False
                    print(new_line + indent + "******* See you *******" + 2 * new_line)                


    def print_board(self, board, indent):
        board_print = ""
        for x in range(len(board)):
            board_print += 2 * indent + str(board[x]) + "\n"
        return board_print


    def print_q_table(self, q_table, state, indent, answer, new_line):
        q_table_print = answer + "All q_table data at state " + str(state) + ":" + 2*new_line
        for i in range(len(q_table[state,:])):
            q_table_print += indent + str(i) + ".: " + str(q_table[state,i]) + new_line
        
        pos_values, highest_value = self.get_highest_values(q_table, state)
        q_table_print += new_line + answer + "Highest value is " + str(highest_value) + " and can be found at position(s): " + pos_values + 2*new_line

        return q_table_print


    def get_highest_values(self, q_table, state):
        values = []
        highest_value = float('-inf')
        for i in range(len(q_table[state,:])):
            if q_table[state,i] > highest_value:
                highest_value = q_table[state,i]
                values.clear()
                values.append(i)
            elif q_table[state,i] == highest_value:
                values.append(i)
        
        pos_values = ""
        for k in range(len(values)):
            pos_values += str(values[k])
            if k < len(values)-1:
                pos_values += ", "
        return pos_values, highest_value


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