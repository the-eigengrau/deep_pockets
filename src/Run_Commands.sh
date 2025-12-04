Run Commands:

#Baseline
python -m src.baselines


#Experiments
python -m src.run_rl_experiment --agent ppo --timesteps 300000

python -m src.run_rl_experiment --agent sac --timesteps 300000

python -m src.run_rl_experiment --agent td3 --timesteps 300000


#Hyper Param Sweeps

python -m src.sweep_hyperparams --agent ppo --timesteps 200000

python -m src.sweep_hyperparams --agent sac --timesteps 200000

python -m src.sweep_hyperparams --agent sac --timesteps 200000
