import subprocess

def run_experiment(dataset, key):
    # train
    cmd = [
            "py", "train_models.py", 
            "--config", "configs/replicate_results/config.yaml",
            "--wandb",
            "--key", key,
            "--dataset", dataset
        ]  
    print(cmd)
    subprocess.run(cmd)
    # evaluate
    cmd = [
            "py", "evaluate.py", 
            "--config", "configs/replicate_results/fulsang_population.yaml",
            "--key", key,
            "--dataset", dataset
        ]  
    print(cmd)
    subprocess.run(cmd)
    # train ridge
    cmd = [
        "py", "train_ridge.py", 
        "--key", key,
        "--dataset", dataset
    ]  
    print(cmd)
    subprocess.run(cmd)
    # eval_ridge
    cmd = [
        "py", "eval_ridge.py", 
        "--key", key,
        "--dataset", dataset,
        "--wandb"
    ]  
    print(cmd)
    subprocess.run(cmd)

run_experiment('fulsang', 'population')
run_experiment('jaulab', 'population')
run_experiment('skl', 'population')
run_experiment('fulsang', 'subj_specific')
run_experiment('jaulab', 'subj_specific')
run_experiment('skl', 'population')


