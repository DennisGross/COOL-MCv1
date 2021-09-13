class Agent():

    def __init__(self, use_tf_environment = False):
        self.use_tf_environment = use_tf_environment

    def select_action(self, time_step, deploy):
        pass

    def store_experience(self, time_step, action_step, n_time_step):
        pass

    def step_learn(self):
        pass

    def episodic_learn(self):
        pass
    
    def get_hyperparameters(self):
        pass

    def save(self, root_folder):
        pass

    def load(self, root_folder):
        pass


def to_tuple(number_of_elements, values):
    '''
    Creates a tuple with
    :param number_of_elements: number of elements
    :param values: element values
    :return: tuple
    '''
    n_tuple = []
    for i in range(number_of_elements):
        n_tuple.append(values)
    return tuple(n_tuple)

