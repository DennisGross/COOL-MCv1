
import stormpy
import json
import sys
import tf_agents
import time

class ModelChecker():

    def optimal_checking(self, environment, prop):
        '''
        Uses Storm to model check the PRISM environment.
        :environment, it contains the path to the PRISM file and contains the constant defintions
        :prop, property specifications as a string
        :return model checking result
        '''
        constant_definitions = environment.storm_bridge.constant_definitions
        formula_str = prop
        start_time = time.time()
        prism_program = stormpy.parse_prism_program(environment.storm_bridge.path)
        prism_program = stormpy.preprocess_symbolic_input(prism_program, [], constant_definitions)[0].as_prism_program()
        properties = stormpy.parse_properties(formula_str, prism_program)
        options = stormpy.BuilderOptions([p.raw_formula for p in properties])
        options.set_build_state_valuations()
        model = stormpy.build_sparse_model_with_options(prism_program, options)
        model_size = len(model.states)
        start_checking_time = time.time()
        result = stormpy.model_checking(model, properties[0])
        initial_state = model.initial_states[0]
        mdp_reward_result = result.at(initial_state)
        return mdp_reward_result, model_size, (time.time()-start_time), (time.time()-start_checking_time)

    def __get_clean_state_dict(self, state_valuation_json: str, example_json: str) -> dict:
        '''
        Get the clean state dictionary
        :param state_valuation_json: state valuation as json
        :param example_json: example state json
        :return:
        '''
        state_valuation_json = json.loads(str(state_valuation_json))
        state = {}
        # print(state_valuation_json)
        # print(example_json)
        example_json = json.loads(example_json)
        for key in state_valuation_json.keys():
            for _key in example_json.keys():
                if key == _key:
                    state[key] = state_valuation_json[key]
        
        return state

    def __get_action_for_state(self, env, agent, state_dict):
        '''
        Get the action for the current state
        :param env: environment
        :param state_dict: current state
        :param policy: rl policy
        :return: action name
        '''
        state = env.storm_bridge.parse_state(json.dumps(state_dict))
        time_step = env.reset()
        time_step = tf_agents.trajectories.time_step.TimeStep(
            step_type=time_step.step_type, reward=time_step.reward, discount=time_step.discount, observation=state
        )
        action_step= agent.select_action(time_step, True)
        return env.action_mapper.actions[action_step.action]


    def induced_markov_chain(self, agent, env, constant_definitions, formula_str = 'Rmin=? [LRA]'):
        '''
        Creates a markov chain of an MDP induced by a Policy and analyze the policy
        :param agent: agent
        :param prism_file: prism file with the MDP
        :param constant_definitions: constants
        :param property_specification: property specification
        :return: mdp_reward_result, model_size, total_run_time, model_checking_time
        '''
        env.reset()
        self.wrong_choices = 0
        start_time = time.time()
        prism_program = stormpy.parse_prism_program(env.storm_bridge.path)
        suggestions = dict()
        i = 0
        for m in prism_program.modules:
            for c in m.commands:
                if not c.is_labeled:
                    suggestions[c.global_index] = "tau_" + str(i) #str(m.name)
                    i+=1

        prism_program = stormpy.preprocess_symbolic_input(
            prism_program, [], constant_definitions)[0].as_prism_program()
        
       
        prism_program = prism_program.label_unlabelled_commands(suggestions)

        properties = stormpy.parse_properties(formula_str, prism_program)
        options = stormpy.BuilderOptions([p.raw_formula for p in properties])
        #options = stormpy.BuilderOptions()
        options.set_build_state_valuations()
        options.set_build_choice_labels(True)


        def permissive_policy(state_valuation, action_index):
            """
            Whether for the given state and action, the action should be allowed in the model.
            :param state_valuation:
            :param action_index:
            :return: True or False
            """
            simulator.restart(state_valuation)
            available_actions = sorted(simulator.available_actions())
            action_name = prism_program.get_action_name(action_index)
            # conditions on the action
            state = self.__get_clean_state_dict(
                state_valuation.to_json(), env.storm_bridge.state_json_example)
            selected_action = self.__get_action_for_state(env, agent, state)
            # Check if selected action is available.. if not set action to the first available action
            #print(selected_action)
            if len(available_actions) == 0:
                return False
            if (selected_action in available_actions) == False:
                #print(state_valuation.to_json())
                selected_action = available_actions[0]

            cond1 = (action_name == selected_action)
            # print(str(state_valuation.to_json()), action_name)#, state, selected_action, cond1)
            return cond1

        simulator = stormpy.simulator.create_simulator(prism_program, seed=42)
        simulator.set_action_mode(
            stormpy.simulator.SimulatorActionMode.GLOBAL_NAMES)

        constructor = stormpy.make_sparse_model_builder(prism_program, options,
                                                        stormpy.StateValuationFunctionActionMaskDouble(
                                                            permissive_policy))
        model = constructor.build()
        model_size = len(model.states)
        #print(model)
        #print(formula_str)
        properties = stormpy.parse_properties(formula_str, prism_program)
        #print(properties[0])
        model_checking_start_time = time.time()
        result = stormpy.model_checking(model, properties[0])

        initial_state = model.initial_states[0]
        #print('Result for initial state', result.at(initial_state))
        mdp_reward_result = result.at(initial_state)
        return mdp_reward_result, model_size, (time.time()-start_time), (time.time()-model_checking_start_time)