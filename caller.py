import subprocess

cmd = [
    "py", "train_models.py", 
    "--config", 'configs/stim_input/aad_net.yaml',
    "--key", 'population',
    "--dataset", 'fulsang',
    "--wandb",
]  
print(cmd)
subprocess.run(cmd)

cmd = [
    "py", "train_models.py", 
    "--config", 'configs/stim_input/aad_net.yaml',
    "--key", 'subj_specific',
    "--dataset", 'fulsang',
    "--wandb",
]  
print(cmd)
subprocess.run(cmd)