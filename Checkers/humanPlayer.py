from simulation import *
import numpy as np
import h5py


class HumanPlayer:
    def __init__(self, size):
        self.size = size
        self.agents = []
        self.agents_data = []
        self.q_table = 0
        self.programming_running = True
        self.user_keeps_playing = False
        self.game_over = False


    def start(self, board, simulation_settings, logging_settings, agent_save_path):
        game = board
        self.q_table = np.zeros((game.state_space(), game.action_space()))
        
        board_size = simulation_settings[0]
        learning_rate = simulation_settings[4]
        discount_rate = simulation_settings[5]

        advised_learning_enabled = False
        print_advised_learning_results = False
        
        new_line = "\n"
        indent = "         "        
        question = indent + "¿¿¿ "
        answer = indent + "=====>  "
        game_started = 2 * new_line + indent + "******* GAME STARTED *******" + 2 * new_line + self.print_board(game.getBoard(), indent)

        print(self.print_programm_started(new_line))

        while self.programming_running:
            action = input(question + "What do you want to do?" + 2 * new_line)
            print(new_line)

            if action == "new":
                self.q_table = np.zeros((game.state_space(), game.action_space()))
                print(answer + "New Q-Table created!" + 2 * new_line)

            elif action == "agents":
                self.view_agents(agent_save_path, answer, new_line)

            elif action == "train":
                self.train_new_agents(board, simulation_settings, logging_settings)
                self.show_results(indent, answer, new_line)

            elif action == "results":
                self.show_results(indent, answer, new_line)

            elif action == "save current":
                self.save_agent(agent_save_path, False, answer, new_line)

            elif "load" in action or "save" in action or "delete" in action or "use" in action:
                digit = [int(word) for word in action.split() if word.isdigit()]    # Thx to Srikar Appalaraju: https://stackoverflow.com/questions/16009861/get-digits-from-string
                if len(digit) == 0:
                    action = self.no_digits_found(answer, new_line)
                else:
                    digit = digit[0] - 1
                    if "load" in action:
                        self.q_table = self.load_agent(agent_save_path, digit, answer, new_line, True)
                    elif "save" in action:
                        self.save_agent(agent_save_path, digit, answer, new_line)
                    elif "delete" in action:
                        self.delete_agent(agent_save_path, digit, answer, new_line)
                    elif "use" in action:
                        self.use_trained_agent(digit, answer, new_line)


            elif action == "play":
                self.user_keeps_playing = True
                self.game_over = False


            elif action == "close":
                self.close_programm(indent, new_line)


            while self.user_keeps_playing:
                print(game_started)
                state = game.reset()

                if self.game_over:
                    action = input(new_line + question + "Another round?" + indent + "(yes/no/close)" + new_line)

                    if action == "yes":
                        self.game_over = False
                        print(game_started)

                    elif action == "no":
                        self.user_keeps_playing = False
                        print(new_line)

                    elif action == "close":
                        self.close_programm(indent, new_line)


                while not(self.game_over):
                    action = input(question + "How do you want to play this?" + 2 * new_line)
                    print("")                

                    if action == "on":
                            advised_learning_enabled = True
                            print(answer + "All further moves will be used to improve the agent." + 2 * new_line)

                    elif action == "off":
                        advised_learning_enabled = False
                        print(answer + "Agent will not learn from further moves." + 2 * new_line)
                    
                    elif action == "state":
                        print(answer + "Current state: " + str(game.getState()) + 2 * new_line)

                    elif action == "tip":
                        pos_values, _ = self.get_highest_values(self.q_table, state)
                        print(answer + "Agent suggests: " + pos_values + 2 * new_line)

                    elif action == "table":
                            reward = np.argmax(self.q_table[state,:])
                            print(self.print_q_table(self.q_table, state, indent, answer, new_line))               

                    elif action == "print":
                        print_advised_learning_results = True
                        print(answer + "Reward and state will be printed now." + 2 * new_line)

                    elif "check" in action:
                        action = [int(word) for word in action.split() if word.isdigit()]    # Thx to Srikar Appalaraju: https://stackoverflow.com/questions/16009861/get-digits-from-string
                        if len(action) == 0:
                            action = self.no_digits_found(answer, new_line)
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

                    elif action == "cheat":
                        for i in range(board_size + 1, game.action_space()):                        
                            _, _, _, info = game.step(i, False)
                            if info[0] == True:
                                print(answer + "A valid move will be: " + str(i) + 2 * new_line)
                                break

                    elif action == "print off":
                        print_advised_learning_results = False
                        print(answer + "Reward and state will NOT be printed." + 2 * new_line)

                    elif action == "board":
                        print(self.print_board(game.getBoard(), indent))

                    elif action == "stop":
                        self.game_over = True
                        self.user_keeps_playing = False
                        print(indent + "------ Match aborted ------" + 2 * new_line)

                    elif action == "close":
                        self.close_programm(indent, new_line)

                    elif action == "pass":
                            action = str(np.argmax(self.q_table[state,:]))
                            print(answer + "Agent's step: " + action + 2 * new_line)
                    
                    if action.isdigit():
                        action = int(action)

                        if action <= game.action_space():
                            new_state, reward, self.game_over, info = game.step(action)

                            if advised_learning_enabled:    # Update Q-table
                                self.q_table[state, action] = self.q_table[state, action] * (1 - learning_rate) + \
                                learning_rate * (reward + discount_rate * np.max(self.q_table[new_state, :]))

                            if print_advised_learning_results:
                                print(answer + "Reward: " + str(reward) + " || State: " + str(state) + new_line)

                            state = new_state       # Set new state
                            
                            print(self.print_board(game.getBoard(), indent))
                            print(indent + self.move_message(info))



    def train_new_agents(self, board, simulation_settings, logging_settings):
        num_episodes = simulation_settings[1]
        for simulation_episode in range(num_episodes):
            simulation = Simulation()
            simulation.run(board, simulation_settings, logging_settings, str(simulation_episode + 1))
            self.agents.append(simulation.getAgent())
            self.agents_data.append(simulation.get_logging_data())        


    def show_results(self, indent, answer, new_line):
        if len(self.agents_data) == 0:
            print(indent + "No agents were send to the trainings camp, yet." + new_line)
        else:
            print(indent + "Here are the newly trained agents: " + new_line)
            for i in range(len(self.agents_data)):
                print("=====>  " + str(i+1) + ": " + self.agents_data[i][0] + "File: " + self.agents_data[i][1])
        print(new_line)


    def use_trained_agent(self, digit, answer, new_line):
        if digit >= 0 and digit < len(self.agents):
            self.q_table = self.agents[digit]
        else:
            print(answer + "This agent does NOT exist!" + 2*new_line)


    def view_agents(self, agent_save_path, answer, new_line):
        with h5py.File(agent_save_path, "r") as hdf:
            agent_count = len(list(hdf.keys()))
            if agent_count == 0:
                print(answer + "No agents have been saved, yet." + new_line)
            else:
                print("Available agents: ")
                for i in range(agent_count):
                    print(answer + str(i+1))
        print(new_line)


    def load_agent(self, agent_save_path, digit, answer, new_line, print_message = False):
        with h5py.File(agent_save_path, "r") as hdf:
            agent_count = len(list(hdf.keys()))
            if digit >= 0 and digit < agent_count:
                loaded_agent = hdf.get(str(digit))
                if print_message:
                    print(answer + "Agent " + str(digit + 1) + " has been loaded." + 2*new_line)
                return np.array(loaded_agent)
            else:
                print(answer + "The agent you are looking for does not exist." + 2*new_line)
                return None

    
    def save_agent(self, agent_save_path, digit, answer, new_line):
        temp_agents = []
        message = answer

        with h5py.File(agent_save_path, "r") as hdf:        
            agent_count = len(list(hdf.keys()))
            for i in range(agent_count):
                temp_agents.append(self.load_agent(agent_save_path, i, answer, new_line))

        if isinstance(digit, bool):
            temp_agents.append(self.q_table)
            message += "The currently used agent has been saved." + 2*new_line
        elif digit >= 0 and digit < len(self.agents):
            temp_agents.append(self.agents[digit])
            message += "The trained agent " + str(digit + 1) + " has been saved." + 2*new_line
        else:
            message += "The agent you wanted to save does not exist." + 2*new_line

        with h5py.File(agent_save_path, "w") as hdf:
            for k in range(len(temp_agents)):
                hdf.create_dataset(str(k), data = temp_agents[k])
        print(message)


    def delete_agent(self, agent_save_path, digit, answer, new_line):
        temp_agents = []
        message = answer

        with h5py.File(agent_save_path, "r") as hdf:        
            agent_count = len(list(hdf.keys()))
            for i in range(agent_count):
                temp_agents.append(self.load_agent(agent_save_path, i, answer, new_line))
       
        if digit >= 0 and digit < len(temp_agents):
            del temp_agents[digit]
            message += "The agent " + str(digit + 1) + " has been deleted." + 2*new_line
        else:
            message += "The agent you wanted to delete does not exist." + 2*new_line

        with h5py.File(agent_save_path, "w") as hdf:
            for k in range(len(temp_agents)):
                hdf.create_dataset(str(k), data = temp_agents[k])
        print(message)


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


    def close_programm(self, indent, new_line):
        self.programming_running = self.user_keeps_playing = self.game_over = False
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


    def no_digits_found(self, answer, new_line):
        return answer + "No digits found." + 2 * new_line

    
    def print_programm_started(self, new_line):
        print(new_line)
        print("                  ***********************************************" + new_line)
        print("                  ***** WELCOME TO MY REINFORCEMENT PROJECT *****" + new_line)
        print("                  ***********************************************" + 2*new_line)