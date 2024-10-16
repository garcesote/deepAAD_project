import subprocess
# train
# cmd = [
#         "py", "train_models.py", 
#         "--config", "configs/replicate_results/train_vlaai.yaml",
#         "--wandb",
#         "--key", 'subj_specific',
#         "--dataset", ''
#     ]  
# print(cmd)
# subprocess.run(cmd)
def run_experiment(dataset, key, train=True, eval=True, train_ridge=True, eval_ridge=True):
    # train
    cmd = [
            "py", "train_models.py", 
            "--config", "configs/replicate_results/config.yaml",
            "--wandb",
            "--key", key,
            "--dataset", dataset
        ]  
    print(cmd)
    if train: subprocess.run(cmd)
    # eval
    cmd = [
            "py", "evaluate.py", 
            "--config", "configs/replicate_results/config.yaml",
            "--wandb",
            "--key", key,
            "--dataset", dataset
        ]  
    print(cmd)
    if eval: subprocess.run(cmd)
    
    cmd = [
        "py", "train_ridge.py",
        "--wandb", 
        "--key", key,
        "--dataset", dataset
    ]  
    print(cmd)
    if train_ridge: subprocess.run(cmd) 
    # eval_ridge
    cmd = [
        "py", "eval_ridge.py", 
        "--key", key,
        "--dataset", dataset,
        "--wandb"
    ]  
    print(cmd)
    if eval_ridge: subprocess.run(cmd)

# run_experiment('fulsang', 'population', train=False, train_ridge=False)
# run_experiment('jaulab', 'population', train=False, train_ridge=False, eval_ridge=False)
# run_experiment('skl', 'population')
run_experiment('fulsang', 'subj_specific', train=False, train_ridge=False, eval_ridge=False)
# run_experiment('jaulab', 'subj_specific')
# run_experiment('skl', 'population')


