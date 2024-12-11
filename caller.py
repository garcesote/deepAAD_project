import subprocess

cmd = [
    "py", "train_models.py", 
    "--config", "configs/spatial_audio/eval_mesd_dnn.yaml",
    "--key", 'subj_specific',
    "--dataset", 'fulsang',
    "--wandb"
]  
print(cmd)
subprocess.run(cmd)

cmd = [
    "py", "train_models.py", 
    "--config", "configs/spatial_audio/dnn_models.yaml",
    "--key", 'subj_specific',
    "--dataset", 'fulsang',
    "--wandb"
]  
print(cmd)

subprocess.run(cmd)

cmd = [
    "py", "evaluate.py", 
    "--config", "configs/spatial_audio/eval_mesd_dnn.yaml",
    "--key", 'subj_specific',
    "--dataset", 'fulsang',
    "--wandb"
]  
print(cmd)
subprocess.run(cmd)

cmd = [
    "py", "evaluate.py", 
    "--config", "configs/spatial_audio/dnn_models.yaml",
    "--key", 'subj_specific',
    "--dataset", 'fulsang',
    "--wandb"
]  
print(cmd)
subprocess.run(cmd)

cmd = [
    "py", "finetune.py", 
    "--config", "configs/spatial_audio/eval_mesd_dnn.yaml",
    "--dataset", 'fulsang',
    "--wandb"
]  
print(cmd)
subprocess.run(cmd)

cmd = [
    "py", "finetune.py", 
    "--config", "configs/spatial_audio/dnn_models.yaml",
    "--dataset", 'fulsang',
    "--wandb"
]  
print(cmd)
subprocess.run(cmd)