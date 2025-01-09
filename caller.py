import subprocess

# cmd = [
#     "py", "spatial_classifier.py", 
#     "--config", "configs/spatial_audio/diff_mse_criterion.yaml",
#     "--key", 'population',
#     "--dataset", 'fulsang',
#     "--wandb",
# ]  
# print(cmd)
# subprocess.run(cmd)

cmd = [
    "py", "train_models.py", 
    "--config", "configs/spatial_audio/batch_size.yaml",
    "--key", 'population',
    "--dataset", 'fulsang',
    "--wandb",
]  
print(cmd)
subprocess.run(cmd)

cmd = [
    "py", "spatial_classifier.py", 
    "--config", "configs/spatial_audio/batch_size.yaml",
    "--key", 'population',
    "--dataset", 'fulsang',
    "--wandb",
]  
print(cmd)
subprocess.run(cmd)