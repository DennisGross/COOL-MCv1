# Taxi
'''
python3.8 cool_mc.py --task training --prism_file_path taxi_distance_reward.prism --project_name exp09_taxi --max_steps 100 --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=2]' --prop_type "max_prop"
python3.8 cool_mc.py --task decision_tree --prism_file_path taxi_distance_reward.prism --project_name exp09_taxi --max_steps 100 --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=2]'
python3.8 cool_mc.py --task dt_model_checking --prism_file_path taxi_distance_reward.prism --project_name exp09_taxi --max_steps 100 --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=2]'
python3.8 cool_mc.py --task storm_model_checking --prism_file_path taxi_distance_reward.prism --project_name exp09_taxi --max_steps 100 --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=2]'
python3.8 cool_mc.py --task rl_model_checking --prism_file_path taxi_distance_reward.prism --project_name exp09_taxi --max_steps 100 --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=2]'
python3.8 cool_mc.py --task dt_model_checking --prism_file_path taxi_distance_reward.prism --project_name exp09_taxi --max_steps 100  --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=10" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=10]'
python3.8 cool_mc.py --task rl_model_checking --prism_file_path taxi_distance_reward.prism --project_name exp09_taxi --max_steps 100  --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=10" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=10]'
python3.8 cool_mc.py --task storm_model_checking --prism_file_path taxi_distance_reward.prism --project_name exp09_taxi --max_steps 100  --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=10" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=10]'
#python3.8 cool_mc.py --task rl_model_checking --prism_file_path taxi_distance_reward.prism --project_name exp09_taxi --max_steps 100  --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=10000" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=10000]'
# Next command gets killed on our machine (RUN OUT OF MEMORY)
#python3.8 cool_mc.py --task storm_model_checking --prism_file_path taxi_distance_reward.prism --project_name exp09_taxi --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=10000]'
# Compare the following result
python3.8 cool_mc.py --task rl_model_checking --prism_file_path taxi_distance_reward.prism --project_name exp09_taxi --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=100" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=100]'
# with
python3.8 cool_mc.py --task training --prism_file_path taxi_distance_reward.prism --project_name exp10_taxi_dis --max_steps 100 --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=100]' --disabled_features jobs_done

python3.8 cool_mc.py --task decision_tree --prism_file_path taxi_distance_reward.prism --project_name exp10_taxi_dis --max_steps 100 --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=100]' --disabled_features jobs_done
python3.8 cool_mc.py --task dt_model_checking --prism_file_path taxi_distance_reward.prism --project_name exp10_taxi_dis --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=7500]' --disabled_features jobs_done
python3.8 cool_mc.py --task rl_model_checking --prism_file_path taxi_distance_reward.prism --project_name exp10_taxi_dis --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=7500]' --disabled_features jobs_done
python3.8 cool_mc.py --task storm_model_checking --prism_file_path taxi_distance_reward.prism --project_name exp10_taxi_dis --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=7500]' --disabled_features jobs_done
# Random Policy
python3.8 cool_mc.py --task rl_model_checking --prism_file_path taxi_distance_reward.prism --project_name exp09_taxi_random --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2" --architecture random --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F jobs_done=100]'
'''
python3.8 cool_mc.py --task plot_rewards --prism_file_path taxi_distance_reward.prism --project_name exp09_taxi --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2"
python3.8 cool_mc.py --task plot_props --prism_file_path taxi_distance_reward.prism --project_name exp09_taxi --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2"
python3.8 cool_mc.py --task plot_rewards --prism_file_path taxi_distance_reward.prism --project_name exp10_taxi_dis --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2"
python3.8 cool_mc.py --task plot_props --prism_file_path taxi_distance_reward.prism --project_name exp10_taxi_dis --constant_definitions "passenger_location_x=0,passenger_location_y=4,passenger_destination_x=0,passenger_destination_y=0,MAX_JOBS=2"