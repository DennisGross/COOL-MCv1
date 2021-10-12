#python cool_mc.py --task rl_model_checking --max_steps 20 --prism_file_path frozen_lake_4x4.prism --project_name example1_vice_versa --constant_definitions "slippery=0.04" --architecture dqn --prop 'Pmax=? [F "water"]' --prop_type "min_prop"
python cool_mc.py --task training --max_steps 20 --prism_file_path frozen_lake_4x4.prism --project_name exp04_FL_4x4xxxxx --constant_definitions "slippery=0.04" --architecture sarsamax --num_episodes 10000 --eval_interval 250 --prop 'Pmax=? [F "water"]' --prop_type "min_prop"