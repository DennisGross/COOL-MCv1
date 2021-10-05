python cool_mc.py --task training --max_steps 20 --prism_file_path frozen_lake_4x4.prism --project_name example1 --constant_definitions "slippery=0.04" --architecture dqn --num_episodes 10000 --eval_interval 100 --prop 'Pmax=? [F "water"]' --prop_type "min_prop"
python cool_mc.py --task decision_tree --max_steps 20 --prism_file_path frozen_lake_4x4.prism --project_name example1 --constant_definitions "slippery=0.04" --architecture dqn --prop 'Pmax=? [F "water"]' --prop_type "min_prop"
python cool_mc.py --task dt_model_checking --max_steps 20 --prism_file_path frozen_lake_4x4.prism --project_name example1 --constant_definitions "slippery=0.04" --architecture dqn --prop 'Pmax=? [F "water"]' --prop_type "min_prop"
python cool_mc.py --task rl_model_checking --max_steps 20 --prism_file_path frozen_lake_4x4.prism --project_name example1 --constant_definitions "slippery=0.04" --architecture dqn --prop 'Pmax=? [F "water"]' --prop_type "min_prop"
python cool_mc.py --task storm_model_checking --max_steps 20 --prism_file_path frozen_lake_4x4.prism --project_name example1 --constant_definitions "slippery=0.04" --architecture dqn --prop 'Pmin=? [F "water"]' --prop_type "min_prop"
python cool_mc.py --task attention_training --project_name example1 --constant_definitions "slippery=0.1"
python cool_mc.py --task attention_mapping --project_name example1 --constant_definitions "slippery=0.1" --attention_input "x=0,y=2"
python cool_mc.py --task rl_model_checking --project_name example1 --constant_definitions "slippery=[0.1;0.1;1]" --prop 'Pmin=? [F "water"]'
python cool_mc.py --task plot_rewards --project_name example1 --constant_definitions "slippery=0.04"
python cool_mc.py --task plot_props --project_name example1 --constant_definitions "slippery=0.04"

