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

# cmd = [
#     "py", "spatial_classifier.py", 
#     "--config", "configs/spatial_audio/diff_mae_criterion.yaml",
#     "--key", 'population',
#     "--dataset", 'fulsang',
#     "--wandb",
# ]  
# print(cmd)
# subprocess.run(cmd)

cmd = [
    "py", "spatial_classifier.py", 
    "--config", "configs/spatial_audio/ild_criterion.yaml",
    "--key", 'population',
    "--dataset", 'fulsang',
    "--wandb",
]  
print(cmd)
subprocess.run(cmd)