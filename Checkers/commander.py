from trainer import *
import numpy as np
import h5py


class Commander:
    def __init__(self, size):
        self.size = size
        self.agents = []
        self.agents_data = []
        self.q_table = 0
        self.programming_running = True
        self.user_keeps_playing = False
        self.game_over = False


    def start(self, board, training_settings, logging_settings, agent_save_path):
        game = board
        self.q_table = np.zeros((game.state_space(), game.action_space()))
        
        board_size = training_settings[0]
        learning_rate = training_settings[4]
        discount_rate = training_settings[5]

        advised_learning_enabled = False
        print_advised_learning_results = False
        
        new_line = "\n"
        indent = "         "        
        question = indent + "¿¿¿ "
        answer = indent + "=====>  "
        options = indent + "(options)" + 2 * new_line
        game_started = 2 * new_line + indent + "******* GAME STARTED *******" + 2 * new_line + self.print_board(game.getBoard(), indent)

        self.print_programm_started(new_line)

        while self.programming_running:
            action = input(question + "What do you want to do?" + options)
            print(new_line)

            if action == "options":
                self.show_options(self.get_program_options(), indent, new_line)

            elif action == "new":
                self.q_table = np.zeros((game.state_space(), game.action_space()))
                print(answer + "New agent has been recruited!" + 2 * new_line)

            elif action == "agents":
                self.view_agents(agent_save_path, answer, new_line)

            elif action == "train new":
                self.train_new_agents(board, training_settings, logging_settings)
                self.show_results(indent, answer, new_line)

            elif action == "train current":
                self.train_new_agents(board, training_settings, logging_settings, self.q_table)
                self.show_results(indent, answer, new_line)

            elif action == "results":
                self.show_results(indent, answer, new_line)

            elif action == "clear":
                self.clear_trained_agents(answer, new_line)

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
                    action = input(question + "How do you want to play this?" + options)
                    print("")                

                    if action == "options":
                        self.show_options(self.get_game_options(), indent, new_line)

                    elif action == "learn on":
                            advised_learning_enabled = True
                            print(answer + "All further moves will be used to improve the agent." + 2 * new_line)

                    elif action == "learn off":
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

                    elif action == "print on":
                        print_advised_learning_results = True
                        print(answer + "Reward and state will be printed now." + 2 * new_line)

                    elif action == "print off":
                        print_advised_learning_results = False
                        print(answer + "Reward and state will NOT be printed." + 2 * new_line)

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


    def show_options(self, options, answer, new_line):        
        
        blank = " "
        max_chars_of_explanations_per_line = 71
        max_chars_to_explanations = 20

        print(answer + "Here are all your options:" + new_line)
        for i in range(len(options)):
            number_of_blanks = max_chars_to_explanations - len(options[i][0])
            message = options[i][0] + blank * number_of_blanks

            explanation_word_list = options[i][1].split()
            remaining_chars_per_line = max_chars_of_explanations_per_line

            for k in range(len(explanation_word_list)):
                number_of_chars_in_word = len(explanation_word_list[k]) + 1  # for blanks after each word
                if remaining_chars_per_line - number_of_chars_in_word >= 0:
                    message += explanation_word_list[k] + blank
                    remaining_chars_per_line -= number_of_chars_in_word
                else:
                    message += new_line + max_chars_to_explanations * blank + explanation_word_list[k] + blank
                    remaining_chars_per_line = max_chars_of_explanations_per_line - number_of_chars_in_word
            
            print(message + new_line)
        print(new_line)


    def train_new_agents(self, board, training_settings, logging_settings, q_table = False):
        num_episodes = training_settings[1]
        for training_episode in range(num_episodes):
            trainer = Trainer()
            trainer.run(board, training_settings, logging_settings, str(training_episode + 1), q_table)
            self.agents.append(trainer.getAgent())
            self.agents_data.append(trainer.get_logging_data())        


    def show_results(self, indent, answer, new_line):
        if len(self.agents_data) == 0:
            print(indent + "No agents were send to the trainings camp, yet." + new_line)
        else:
            print(indent + "Here are the newly trained agents: " + new_line)
            for i in range(len(self.agents_data)):
                print("=====>  " + str(i+1) + ": " + self.agents_data[i][0] + "File: " + self.agents_data[i][1])
        print(new_line)


    def clear_trained_agents(self, answer, new_line):
        self.agents = []
        self.agents_data = []
        print(answer + "All trained agents are fired." + 2*new_line)


    def use_trained_agent(self, digit, answer, new_line):
        if digit >= 0 and digit < len(self.agents):
            self.q_table = self.agents[digit]
            print(answer + "Agent " + str(digit + 1) + " is waiting for action!" + 2*new_line)
        else:
            print(answer + "This agent does NOT exist!" + 2*new_line)


    def view_agents(self, agent_save_path, answer, new_line):
        with h5py.File(agent_save_path, "a") as hdf:
            agent_count = len(list(hdf.keys()))
            if agent_count == 0:
                print(answer + "No agents have been saved, yet." + new_line)
            else:
                print("Available agents: ")
                for i in range(agent_count):
                    print(answer + str(i+1))
        print(new_line)


    def load_agent(self, agent_save_path, digit, answer, new_line, print_message = False):
        with h5py.File(agent_save_path, "a") as hdf:
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

        with h5py.File(agent_save_path, "a") as hdf:        
            agent_count = len(list(hdf.keys()))
            for i in range(agent_count):
                temp_agents.append(self.load_agent(agent_save_path, i, answer, new_line))

        if isinstance(digit, bool):
            temp_agents.append(self.q_table)
            message += "The currently used agent has been saved as agent " + str(len(temp_agents)) + "."
        elif digit >= 0 and digit < len(self.agents):
            temp_agents.append(self.agents[digit])
            message += "The trained agent " + str(digit + 1) + " has been saved as agent " + str(len(temp_agents)) + "."
        else:
            message += "The agent you wanted to save does not exist."

        with h5py.File(agent_save_path, "w") as hdf:
            for k in range(len(temp_agents)):
                hdf.create_dataset(str(k), data = temp_agents[k])
        print(message + 2*new_line)


    def delete_agent(self, agent_save_path, digit, answer, new_line):
        temp_agents = []
        message = answer

        with h5py.File(agent_save_path, "a") as hdf:        
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
        print(new_line + indent + "********************* SEE YOU *********************" + 3 * new_line)


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
        print("                  *************************************************" + new_line)
        print("                  ****** WELCOME TO MY REINFORCEMENT PROJECT ******" + new_line)
        print("                  *************************************************" + 2*new_line)


    def get_program_options(self):
        program_options = [
            ["#","At options with a hashtag, you have to replace it with a digit to select the desired agent."],
            ["agents", "Shows you all the saved agents."],
            ["clear","Deleting all the newly trained agents and its results of this session. Logs will not be removed."],
            ["close","Ends the current session and closes the program."],
            ["delete #","Deletes the selected saved agent."],
            ["load #","Loads the selected trained agent and puts that one in charge. Available agents can be seen in 'agents'. All further operations in 'play' will be done on this agent."],
            ["play","Go into play mode. You can use your currently in charge agent and experiement with it."],
            ["results","Shows the results of the trainings of this session."],            
            ["save #","Saves the selected trained agent and can be loaded at another session."],
            ["save current","Saves the currently in charge agent."],
            ["train new", "Trains a new agent by using all the prepared settings."],
            ["train current", "Sends the currently used agent to the training by using all the prepared settings."],
            ["use #","Puts the selected trained agent in charge. Available agents can be seen in 'results'. All further operations in 'play' will be done on this agent. Use 'save current' to save the progress of this agent for another session."]
        ]        
        return program_options


    def get_game_options(self):
        game_options = [
            ["#","Enter a digit to make a step on the board."],
            ["board","Shows you the current state of the board."],
            ["cheat","Gives you a valid step."],
            ["check #","Enter 'check' + a digit to check, if that step would be valid or not."],
            ["close","Ends the current session and closes the program."],
            ["learn off","Currently in charge agent does NOT learn from any steps made from now on. This is the default option, when the program has started."],
            ["learn on","Currently in charge agent learns from any steps made from now on."],            
            ["pass","The agent will make a step, which it believes is the best one."],
            ["print off","At each step, the gained reward and state will NOT be printed, anymore. This is the default option, when the program has started."],
            ["print on","At each step, the gained reward and state will be printed."],
            ["state","Gives you the current state of the board."],
            ["stop","Stops the current game and you get back to the program options."],
            ["table","This prints all entries of the currently in charge agent for the current state."],
            ["tip","The currently in charge agent suggests the best steps it knows about."]
        ]
        return game_options