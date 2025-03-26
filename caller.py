import subprocess

cmd = [
    "py", "train_models.py", 
    "--config", 'configs/euroacustics/best_models.yaml',
    "--key", 'subj_specific',
    "--dataset", 'fulsang',
    "--cross_val",
    "--wandb",
]  
print(cmd)
subprocess.run(cmd)

cmd = [
    "py", "evaluate.py", 
    "--config", 'configs/euroacustics/best_models.yaml',
    "--key", 'subj_specific',
    "--dataset", 'fulsang',
    "--cross_val",
    "--wandb",
]  
print(cmd)
subprocess.run(cmd)