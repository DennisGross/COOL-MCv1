# Comprehensive tOOl for Learning and Model Checking (COOL-MC)
Safety is a major issue of reinforcement learning (RL) in complex real-world scenarios.
In recent years, formal verification has increasingly been used to provide rigorous safety guarantees for RL.
However, until now, major technical obstacles for combining probabilistic model checking with, in particular deep RL, remain.
Our easy-to-use tool-chain COOL-MC unifies the powerful toolset of model checking, interpretable machine learning, and deep RL.
At the heart is a tight integration of learning and verification that involves, amongst others, (1) the incremental building of state spaces, (2) mapping of policies obtained from state-of-the-art deep RL to formal models, and (3) the use of features from interpretable and explainable machine learning such as decision trees and attention maps.
We evaluate our tool-chain on multiple commonly use.

##### Content
1. Getting Started with COOL-MC
2. Example 1 (Frozen Lake)
3. Example 2 (Taxi)
4. General Pipeline
5. RL Agent Training
6. RL Model Checking
7. RL Policy Interpretation
8. RL Policy Decision Explaination
9. Storm Model Checking
10. Command Line Arguments
11. Benchmarking
12. Manual Installation

## Getting Started with COOL-MC (Latest Release)
We assume that you have docker installed and that you run the following commands in the root of this repository:
1. Download the docker container [here](https://drive.google.com/file/d/10C3PkC6uU0M-FEY58zeVOER8CK9zUO3L/view?usp=sharing).
2. Load the docker container: `docker load --input coolmc.tar`
3. Create a project folder: `mkdir projects`
4. Run the docker container: `docker run --user "$(id -u):$(id -g)" -v "$(pwd)/projects":/projects -v "$(pwd)/prism_files":/prism_files -it coolmc bash`

We discuss how to create the docker container yourself, and how to install the tool natively later.

## Example 1 (Frozen Lake)
To demonstrate our tool, we are going to train a near optimal RL policy for the commonly known frozen lake environment.
The goal is to get familar with our tool and being able to use all our supported features.
1. Start the interactive container and mount the PRISM and project folder: Run `docker run --user "$(id -u):$(id -g)" -v "$(pwd)/projects":/projects -v "$(pwd)/prism_files":/prism_files -it coolmc bash`
2. Execute `bash example_1.sh`.

`python cool_mc.py --task training --architecture dqn --max_steps 20 --prism_file_path frozen_lake_4x4.prism --constant_definitions "slippery=0.04" --prop 'Pmax=? [F "water"]' --prop_type "min_prop" --project_name exp01_FL_4x4 --num_episodes 10000 --eval_interval 250`

This first command trains an RL agent to get a near-optimal policy to reach the frisbee without falling into the water. The `task` parameter sets the COOL-MC into training mode (reinforcement learning) and the `architecture` the RL algorithm (in this example: deep q-learning). The `max_steps` parameter is needed to terminate the environment after a given number of steps (20). With the `prism_file_path` parameter, we inform COOL-MC on which environment we want to train our agent. The `constant_definitions` defines the constants of the environment. The `prop` parameter is the property specification, which we want to query while training. `prop_type` that we want to optimize for minimizing the property result. The `project_name` parameter names the project. `num_episodes` defines the number of epochs and `eval_interval` the evaluation and property querying interval.


COOL-MC gives us the possibility to monitor the training progress via Tensorboard (if installed):

`tensorboard --logdir projects # Execute this command on your local machine`. 

After 10000 epochs, we gain an RL policy with a roughly 5% probability of falling into the water. Since the frozen lake environment is quite small, we can check the optimal probability of falling into the water by modifying `task training` to `task storm_model_checking` and removing unnecessary parameters. This allows us to use Storm directly:

`python cool_mc.py --task storm_model_checking --prism_file_path frozen_lake_4x4.prism --constant_definitions "slippery=0.04" --prop 'Pmax=? [F "water"] --project_name exp01_FL_4x4`

By changing `task storm_model_checking` to `task rl_model_checking` we are able to do model checking with the trained RL policy:

`python cool_mc.py --task rl_model_checking --prism_file_path frozen_lake_4x4.prism --constant_definitions "slippery=0.04" --prop 'Pmax=? [F "water"] --project_name exp01_FL_4x4`

By changing `task rl_model_checking` to `task decision_tree` we are able to interprete the trained policy via a decision tree:

`python cool_mc.py --task decision_tree --prism_file_path frozen_lake_4x4.prism --project_name example1 --constant_definitions "slippery=0.04" --architecture dqn --prop 'Pmax=? [F "water"]'`

We can find the decision tree plot inside the project folder.

![Decision Tree](https://github.com/DennisGross/COOL-MC/blob/main/doc/images/decision_tree.png)
*Decision Tree Policy Interpretation of RL policy of example 1.*

After training the decision tree, we can use `task dt_model_checking` to model check the decision tree policy:

`python cool_mc.py --task dt_model_checking --prism_file_path frozen_lake_4x4.prism --project_name example1 --constant_definitions "slippery=0.04"`

By changing `task dt_model_checking` to `task attention_training,` we can train an explainable model for our trained RL policy. Now it is possible to see which features influence the RL policy decision for a certain state:

`python cool_mc.py --task attention_training --project_name example1 --constant_definitions "slippery=0.1"`

`python cool_mc.py --task attention_mapping --project_name example1 --constant_definitions "slippery=0.1" attention_input "x=0,y=2"`

You can find the attention mapping plot in the project folder:

![Attention Map Plot](https://github.com/DennisGross/COOL-MC/blob/main/doc/images/attention_map.png)
*Attention Map of a RL policy.*

If we interested in how our RL agent performs over a range of different environment initializations, we can use the following command:

`python3.8 cool_mc.py --task rl_model_checking --project_name example1 --constant_definitions "slippery=[0.1;0.1;1]" --prop 'Pmin=? [F "water"]'`

![Properties over a range of Constant definition](https://github.com/DennisGross/COOL-MC/blob/main/doc/images/properties.png)

*A plot that visualizes how different constant definitions influence the safety property of the trained RL policy.*

Besides using Tensorboard, it is also possible to directly plot the rewards and property results while training and save them into the project folder:

![Rewards while Training](https://github.com/DennisGross/COOL-MC/blob/main/doc/images/reward_plotting.png)

*`python cool_mc.py --task plot_rewards --project_name example1`*

![Property Results while Training](https://github.com/DennisGross/COOL-MC/blob/main/doc/images/prop_plotting.png)

*`python cool_mc.py --task plot_props --project_name example1`*

## Example 2 (Taxi)
To demonstrate our tool, we are going to train a near optimal RL policy for the commonly known taxi environment.
1. Start the interactive container and mount the PRISM and project folder: Run `docker run --user "$(id -u):$(id -g)" -v "$(pwd)/projects":/projects -v "$(pwd)/prism_files":/prism_files -it coolmc bash`
2. Execute `bash example_2.sh`.

`bash example_2.sh` executes five commands.

The first command trains a DQN policy on the taxi environment and queries every 500 epochs the probability of finishing two jobs.

The second command uses Storm to get the optimal probability of finishing two jobs.

The third command uses the trained DQN policy to get the probability of finishing two jobs.

The fourth command generates an interpretable decision tree policy of the trained DQN policy.

The fifth command uses the interpretable decision tree policy to get the probability of finishing two jobs.

The sixth command trains an attention autoencoder for the explainability of the DQN policy.

With command 7, it is possible to see which state variables influence the DQN decision for the given input (explainable reinforcement learning).


## General Pipeline
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


## Command Line Arguments
The following list contains all the major COOL-MC command line arguments.
It does not contain the arguments which are related to the RL algorithms.
For a detailed description, we refer to the src.rl_agents package.


### Task
`task` specifies the task: 
- Reinforcement Learning (training)
- RL Model Checking (rl_model_checking)
- Storm Model Checking (storm_model_checking)
- Attention Training (attention_training)
- Attention Map Plotting (attention_mapping)
- decision tree training (decision_tree)
- Plot Rewards of the training progress (plot_rewards)
- Plot property results of the training progress (plot_props)

### prism_dir
Specifies the folder with all PRISM files.

### prism_file_path
Specifies the path of the PRISM file inside the prism file folder.

### project_dir
Specifies the folder with the COOL-MC projects.

### project_name
Specifies the current COOL-MC project name and saves it inside the `project_dir` folder.

### constant_definitions
Constant definitions for the environments. Examples:
- x=0.2
- x=0.3,y=1
- x=4,y=3,z=3
- x=[0,3,30],y=3 with a range of x-values from 0 until 27 with step size of 3.

### max_steps
Specifies the maximal number of allowed steps inside the environment.

### reward_training
If true, it disables the property querying.

### reward_flag
If true, the agent receives rewards instead of penalties.

### wrong_action_penalty
Penaly of choosing wrong actions.

### architecture
Specifies the agent algorithm (investigate src.rl_agents.agent_builder for supported agents).

### num_episodes
Specifies the number of learning episodes.

### num_spervised_epochs
Specifies the number of supervised learning episodes (necessary for interpretable/explainable machine learning).

### eval_interval
Specifies the evaluation interval of the RL policy.

### prop
Specifies the property specification.

### prop_type
Optimizes the RL policy for Maximal Reward (max_reward), Minimal Reward (min_reward), Property minimization (min_prop) or maximization (max_prop).

### disabled_features
Features which should not be used by the RL agent: FEATURE1,FEATURE2,...

### attention_input
Input for attention map plotting.

### no_gpu
Disable GPU.


## Benchmarking
We tested our tool on a variety of benchmarks to compare PRISM modeling styles, COOL-MC features, and scalability.
The results of these experiments are described in our paper. To reproduce our results, run the following shell scripts:

- Run `bash experiments_frozen_lake.sh` to reproduce the comparison of multiple RL algorithms in the frozen lake environment.
- Run `bash experiments_taxi.sh` to reproduce the taxi environment experiments.
- Run `bash experiments_smart_grid.sh` to reproduce the smart grid environment experiments.
- Run `bash experiments_qcomp.sh` to verify that the QComp benchmarks with reward functions are working.
- Run `bash experiments_modified_qcomp.sh` to verify that the QComp benchmarks with dummy reward functions are working.

Note: Experiments which RAN OUT OF MEMORY or TIME OUTs are commented out.

## Manual Installation

### Creating the Docker

You can build the container via `docker build -t coolmc .` It is also possible for UNIX users to run the bash script in the bin-folder.

### Installing the tool
Switch to the repository folder and define environment variable `COOL_MC="$PWD"`

#### (1) Install Dependencies
`sudo apt-get update && sudo apt-get -y install build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev python3 python-is-python3 python3-setuptools python3-pip graphviz && sudo apt-get install -y --no-install-recommends maven uuid-dev virtualenv`

#### (2) Install Storm
0. `cd $COOL_MC`
1. `git clone https://github.com/moves-rwth/storm.git`
2. `cd storm`
3. `mkdir build`
4. `cd build`
5. `cmake ..`
6. `make -j 1`

For more information about building Storm, click [here](https://www.stormchecker.org/documentation/obtain-storm/build.html).

For testing the installation, follow the follow steps [here](https://www.stormchecker.org/documentation/obtain-storm/build.html#test-step-optional).

#### (3) Install PyCarl
0. `cd $COOL_MC`
1. `git clone https://github.com/moves-rwth/pycarl.git`
2. `cd pycarl`
3. `python setup.py build_ext --jobs 1 develop`

If permission problems: `sudo chmod 777 /usr/local/lib/python3.8/dist-packages/` and run third command again.


#### (4) Install Stormpy

0. `cd $COOL_MC`
1. `git clone https://github.com/moves-rwth/stormpy.git`
2. `cd stormpy`
3. `python setup.py build_ext --storm-dir "${COOL_MC}/build/" --jobs 1 develop`

For more information about the Stormpy installation, click [here](https://moves-rwth.github.io/stormpy/installation.html#installation-steps).

For testing the installation, follow the steps [here](https://moves-rwth.github.io/stormpy/installation.html#testing-stormpy-installation).

#### (5) Install remaining python packages and create project folder
0. `cd $COOL_MC`
1. `pip install -r requirements.txt`
2. `mkdir projects`

