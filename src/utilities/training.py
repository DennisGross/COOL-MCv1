import math

from src.utilities.reporter import Reporter

def train(project, monitor, num_episodes, num_evals):
    '''
    Trains the agent of a given project over num_episodes episodes and evaluates the trained policy every num_eval episodes.
    :param project, given project
    :param monitor, monitors the training process
    :param num_episodes, number of training episodes
    :param num_evals, step size for evaluation
    '''
    avg_return = -math.inf
    prop_result = None
    reporter = Reporter(project.report)
    for episode in range(num_episodes):
        if project.agent.use_tf_environment:
            time_step = project.environment.tf_environment.reset()
        else:
            time_step = project.environment.reset()
        while not time_step.is_last():
            project.data_collector.store_time_step(time_step)
            if project.agent.use_tf_environment:
                action_step = project.agent.select_action(time_step, False)
                n_time_step = project.environment.tf_environment.step(action_step.action)
            else:
                action_step = project.agent.select_action(time_step, False)
                n_time_step = project.environment.step(action_step.action)
            project.agent.store_experience(time_step, action_step, n_time_step)
            project.agent.step_learn()
            time_step = n_time_step
        project.agent.episodic_learn()
        
        avg_return, all_returns = average_return(project)
        if episode % project.report['eval_interval'] == 0 and episode != 0:
            prop_result = None
            model_size = None
            model_checking_time = None
            checking_time = None
            if project.report['reward_training'] == False:
                prop_result, model_size, model_checking_time, checking_time = project.environment.storm_bridge.model_checker.induced_markov_chain(project.agent, project.environment, project.environment.storm_bridge.constant_definitions, project.report['prop'])
            if project.report['prop_result'] == None or project.report['prop_type'] == 'min_prop' and prop_result < project.report['prop_result'] or project.report['prop_type'] == 'max_prop' and prop_result >project.report['prop_result'] or project.report['prop_type'] == 'max_reward' and max(all_returns) >= project.report['return'] or project.report['prop_type'] == 'min_reward' and min(all_returns) <= project.report['return']:
                reporter.write_best_results(prop_result, all_returns, model_size, model_checking_time, episode, checking_time, project.report['prop_type'])
                # Save Agent
                project.save_agent()
            # Additionally: Save all results
            reporter.write_results(prop_result, avg_return, model_size, model_checking_time, checking_time)
            project.save_report_and_data()
            
        monitor.monitor(episode, avg_return, project.report['return'], prop_result)
        
            
            
        

def average_return(project, num_episodes=1):
    '''
    This function calculates the average return over num_episodes episodes.
    :param num_episode: number of episodes
    :return: average return
    '''
    all_returns = []
    for episode in range(num_episodes):
        time_step = project.environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = project.agent.select_action(time_step, True)
            n_time_step = project.environment.step(action_step.action)
            episode_return += float(n_time_step.reward)
            time_step = n_time_step
        all_returns.append(episode_return)
    return sum(all_returns)/num_episodes, all_returns        