import subprocess


cmd = [
    "py", "finetune.py", 
    "--config", 'configs/euroacustics/best_models.yaml',
    "--key", 'population',
    "--dataset", 'fulsang',
    "--wandb",
    "--cross_val",
]  
print(cmd)
subprocess.run(cmd)

cmd = [
    "py", "evaluate.py", 
    "--config", 'configs/euroacustics/best_models.yaml',
    "--key", 'population',
    "--dataset", 'fulsang',
    "--finetuned",
    "--cross_val",
    "--wandb",
]  
print(cmd)
subprocess.run(cmd)

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

cmd = [
    "py", "train_models.py", 
    "--config", 'configs/euroacustics/best_models.yaml',
    "--key", 'subj_independent',
    "--dataset", 'fulsang',
    "--cross_val",
    "--wandb",
]  
print(cmd)
subprocess.run(cmd)

cmd = [
    "py", "evaluate.py", 
    "--config", 'configs/euroacustics/best_models.yaml',
    "--key", 'subj_independent',
    "--dataset", 'fulsang',
    "--cross_val",
    "--wandb",
]  
print(cmd)
subprocess.run(cmd)