from numpy.core.fromnumeric import mean


class Reporter:

    def __init__(self, report):
        '''
        Initialize the reporter.
        :param report, reference on report
        '''
        self.report = report

    def write_best_results(self, prop_result, all_returns, model_size, model_checking_time, episode, checking_time, prop_type):
        '''
        Write the best results into the report
        :prop_result, property result
        :all_returns, all returns
        :model_size, size of the MDP model
        :model_checking_time, the time to build AND model check
        :episode, the current episode
        :checking_time, the model checking time
        :prop_type, property saving type to find best result
        '''
        if prop_type == 'min_reward':
            self.report['return'] = min(all_returns)
        elif prop_type == 'max_reward':
            self.report['return'] = max(all_returns)
        else:
            self.report['return'] = mean(all_returns)
        self.report['episode'] = episode
        self.report['prop_result'] = prop_result
        self.report['model_size'] = model_size
        self.report['model_checking_time'] = model_checking_time
        self.report['checking_time'] = checking_time

    
    def write_results(self,prop_result, reward, model_size, model_checking_time, checking_time):
        '''
        Write the results into the report
        :prop_result, property result
        :reward, reward
        :model_size, size of the MDP model
        :model_checking_time, the time to build AND model check
        :checking_time, the model checking time
        '''
        if 'prop_results' in self.report.keys():
            self.report['prop_results'].append(prop_result)
        else:
            self.report['prop_results'] = [prop_result]

        if 'rewards' in self.report.keys():
            self.report['rewards'].append(reward)
        else:
            self.report['rewards'] = [reward]

        if 'model_sizes' in self.report.keys():
            self.report['model_sizes'].append(model_size)
        else:
            self.report['model_sizes'] = [model_size]

        if 'model_checking_times' in self.report.keys():
            self.report['model_checking_times'].append(model_checking_time)
        else:
            self.report['model_checking_times'] = [model_checking_time]

        if 'checking_times' in self.report.keys():
            self.report['checking_times'].append(checking_time)
        else:
            self.report['checking_times'] = [checking_time]