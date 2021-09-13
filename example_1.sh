python3.8 cool_mc.py --task training --prism_file_path frozen_lake_4x4.prism --project_name example2 --constant_definitions "slippery=0.1" --architecture dqn --num_episodes 10000 --eval_interval 500 --prop 'Pmax=? [F "water"]' --prop_type "min_prop"
python3.8 cool_mc.py --task storm_model_checking --project_name example2 --constant_definitions "slippery=0.1" --prop 'Pmax=? [F "water"]'
python3.8 cool_mc.py --task rl_model_checking --project_name example2 --constant_definitions "slippery=0.1" --prop 'Pmax=? [F "water"]'
python3.8 cool_mc.py --task decision_tree --project_name example2 --constant_definitions "slippery=0.1" --prop 'Pmax=? [F "water"]'
python3.8 cool_mc.py --task attention_training --project_name example2 --constant_definitions "slippery=0.1"
python3.8 cool_mc.py --task attention_mapping --project_name example2 --constant_definitions "slippery=0.1" attention_input "x=0,y=2"