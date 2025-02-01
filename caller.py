import subprocess

# cmd = [
#     "py", "train_models.py", 
#     "--config", 'configs/euroacustics/cnn.yaml',
#     "--key", 'population',
#     "--dataset", 'fulsang',
#     "--cross_val",
#     "--wandb",
# ]  
# print(cmd)
# subprocess.run(cmd)

# cmd = [
#     "py", "evaluate.py", 
#     "--config", 'configs/euroacustics/cnn.yaml',
#     "--key", 'population',
#     "--dataset", 'fulsang',
#     "--cross_val",
#     "--wandb",
# ]  
# print(cmd)
# subprocess.run(cmd)

# cmd = [
#     "py", "finetune.py", 
#     "--config", 'configs/euroacustics/cnn.yaml',
#     "--key", 'population',
#     "--dataset", 'fulsang',
#     "--cross_val",
#     "--wandb",
# ]  
# print(cmd)
# subprocess.run(cmd)

# cmd = [
#     "py", "evaluate.py", 
#     "--config", 'configs/euroacustics/cnn.yaml',
#     "--key", 'population',
#     "--dataset", 'fulsang',
#     "--finetuned",
#     "--cross_val",
#     "--wandb",
# ]  
# print(cmd)
# subprocess.run(cmd)

cmd = [
    "py", "train_models.py", 
    "--config", 'configs/euroacustics/cnn.yaml',
    "--key", 'subj_specific',
    "--dataset", 'fulsang',
    "--cross_val",
    "--wandb",
]  
print(cmd)
subprocess.run(cmd)

cmd = [
    "py", "evaluate.py", 
    "--config", 'configs/euroacustics/cnn.yaml',
    "--key", 'subj_specific',
    "--dataset", 'fulsang',
    "--cross_val",
    "--wandb",
]  
print(cmd)
subprocess.run(cmd)