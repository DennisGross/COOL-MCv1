# Comprehensive tOOl for Learning and Model Checking (COOL-MC)
Safety is a major issue of reinforcement learning (RL) in complex real-world scenarios.
In recent years, formal verification has increasingly been used to provide rigorous safety guarantees for RL.
However, until now, major technical obstacles for combining probabilistic model checking with, in particular deep RL, remain.
Our easy-to-use tool-chain COOL-MC unifies the powerful toolset of model checking, interpretable machine learning, and deep RL.
At the heart is a tight integration of learning and verification that involves, amongst others, (1) the incremental building of state spaces, (2) mapping of policies obtained from state-of-the-art deep RL to formal models, and (3) the use of features from interpretable and explainable machine learning such as decision trees and attention maps.
We evaluate our tool-chain on multiple commonly use

## Build the Docker
Build the container via `docker build -t coolmc .`


## First Steps
We first have to model our RL environment. COOL-MC supports PRISM as modeling language.
It can be difficult to design own PRISM environments.
Here are some tips how to make sure that your PRISM environment works correctly with COOL-MC:
- Make sure that you only use transition-rewards
- After the agent reaches a terminal state, the storm simulator stops the simulation. Therefore, terminal state transitions will not executed. So, do not use self-looping terminal states.
- To improve the training performance, try to make all actions at every state available. Otherwise, the agent may chooses a not available action and receives a penalty.
- Try to unit test your PRISM environment before RL training. Does it behave as you want?

## RL Agent Training
After we modeled the environment, we can train RL agents on this environment.
It is also possible to develop your own RL agents:
1. Create a AGENT_NAME.py in the src.rl_agents package
2. Create a class AGENT_NAME and inherit all methods from src.agent.Agent
3. Set use_tf_environment to true if you use tf_environments instead of py_environments from tf_agents
4. Override all the needed methods (depends on your agent) + the agent save- and load-method.
5. In src.rl_agents.agent_builder extends the build_agent method with an additional elif branch for your agent
6. Add additional command-line arguments in cool_mc.py (if needed)

Here are some tips that may improve the training progress:

- Try to use the disable_state parameter to disable state variables from PRISM which are only relevant for the PRISM environment architecture.
- Play around with the RL parameters.
- Think especially about the size of the max_steps if you have only one terminal state and if this terminal state is a  negative outcome with a huge penalty. Too large max_steps values lead to always reaching this negative step in the beginning of the training. Instead, decrease the max_steps so that the RL agent may not reach this terminal state and stops training earlier. So the huge penality is not influencing the RL training too much in the beginning.
- The model checking part while RL training can take time. Therefore, the best way to train and verify your model is to first use reward_max. After the RL model may reaches an execptable reward the change the parameter prop_type to min_prop or max_prop and adjust the evaluation intervals.



## RL Model Checking
We can use the trained RL policy to create a induce Discrete-Time Markov Chain (DTMC).
This allows us to verify the RL policy in a much smaller DTMC as with Storm.


## RL Policy Interpretation
Another toolset that helps us to make machine learning modelssafer is called interpretable machine learning. It refers to methods and modelsthat make the behavior and predictions of machine learning systems understand-able. We make our trained RL policy interpretable by training a decision treeas  surrogate  model.

## RL Policy Decision Explaination
Humans  want  to  understand  adecision or at least they want to get an explanation for certain decisions. There-fore, COOL-MC also supports attention mapping which is a explainable machinelearning method. For that, we first train an auto encoder with attention lay-ers as a surrogate model of the trained RL policy

## Storm Model Checking
If possible,we also give the user the possibility to calculate the optimal policy of the PRISMenvironment via Storm.
## Example 1 (Frozen Lake)
To demonstrate our tool, we are going to train a near optimal RL policy for the commonly known frozen lake environment.
1. Create via `mkdir projects` the empty project folder in the root of COOL-MC.
2. Start the interactive container and mount the PRISM and project folder: Run `docker run -v "$(pwd)/projects":/projects -v "$(pwd)/prism_files":/prism_files -it coolmc bash`
3. Execute `bash example_1.sh`.

`bash example_1.sh` executes five commands. 

The first command trains a RL agent to get a near optimal policy.
It contains the most important command arguments for the execution of the RL training process.
- `task training` specifies the task.
- `prism_file_path frozen_lake_4x4.prism` specifies the environment.
- `constant_definitions "slippery=0.1"` specifies the constants for the environment.
- `project_name example1` specifies the name of the project. 
- `architecture dqn` specifies the RL algorithm
- `num_episodes 10000` specifies the training epochs
- `eval_interval 500` specifies the evaluation epoch intervals
- `prop 'Pmax=? [F "water"]'` specifies the property specification which we want to check while training.
- `prop_type "min_prop"` specifies if we want to save policies that minimize/maximize the property specification or policies that minimize/maximize rewards.

COOL-MC gives us the possibility to monitor the training progress via tensorboard:
`tensorboard --logdir projects # Execute this command on your local machine`. 
After 10000 epochs, we gain a RL policy with roughly 5% probability of falling into the water. Since the frozen lake environment is quite small, we can check the optimal probability of falling into the water by modifying `task training` to `task storm_model_checking` (second command).

By changing `task storm_model_checking` to `task rl_model_checking` we are able to model check the trained RL policy (third command).

By changing `task rl_model_checking` to `task decision_tree` we are able to interprete the trained policy via a decision tree (fourth command).

By changing `task rl_model_checking` to `task decision_tree` we are able to interprete the trained policy via a decision tree (fourth command). We can find the decision tree plot inside the project folder.

By changing `task decision_tree` to `task attention_training` we are able to train an explainable model for our trained RL policy. Now it is possible to see which features influence the RL policy decision for a certain state. You can find the attention mapping plot in the project folder.


## Example 2 (Taxi)
To demonstrate our tool, we are going to train a near optimal RL policy for the commonly known taxi environment.
1. Create via `mkdir projects` the empty project folder in the root of COOL-MC.
2. Start the interactive container and mount the PRISM and project folder: Run `docker run -v "$(pwd)/projects":/projects -v "$(pwd)/prism_files":/prism_files -it coolmc bash`
3. Execute `bash example_2.sh`.

`bash example_2.sh` executes five commands.

It contains the following command arguments for the execution of the RL training process.
- `task training` specifies the task.
- `prism_file_path taxi_distance_reward.prism` specifies the environment.
- `constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2"` specifies the constants for the environment.
- `project_name example2` specifies the name of the project. 
- `architecture dqn` specifies the RL algorithm
- `num_episodes 10000` specifies the training epochs
- `eval_interval 500` specifies the evaluation epoch intervals
- `prop 'Pmax=? [F jobs_done=2]'` specifies the property specification which we want to check while training.
- `prop_type "max_prop"` specifies if we want to save policies that minimize/maximize the property specification or policies that minimize/maximize rewards.



