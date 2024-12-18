import subprocess

cmd = [
    "py", "train_models.py", 
    "--config", "configs/spatial_audio/diff_mse_criterion.yaml",
    "--key", 'subj_specific',
    "--dataset", 'fulsang',
    "--wandb",
]  
print(cmd)
subprocess.run(cmd)

cmd = [
    "py", "train_models.py", 
    "--config", "configs/spatial_audio/diff_mse_criterion.yaml",
    "--key", 'population',
    "--dataset", 'fulsang',
    "--wandb",
]  
print(cmd)
subprocess.run(cmd)