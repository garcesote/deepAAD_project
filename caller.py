import subprocess

# datasets
# datasets = ['kuleuven', 'jaulab']
datasets = ['kuleuven']

for dataset in datasets:

    # cmd = [
    #     "py", "train_models.py", 
    #     "--config", 'configs/dataset_comparison/aad_net.yaml',
    #     "--key", 'population',
    #     "--dataset", dataset,
    #     "--project", 'dataset_comparison',
    #     "--cross_val",
    #     "--wandb",
    # ]  
    # print(cmd)
    # subprocess.run(cmd)

    cmd = [
        "py", "finetune.py", 
        "--config", 'configs/dataset_comparison/aad_net.yaml',
        "--key", 'population',
        "--dataset", dataset,
        "--project", 'dataset_comparison',
        "--cross_val",
        "--wandb",
    ]  
    print(cmd)
    subprocess.run(cmd)

    cmd = [
        "py", "evaluate.py", 
        "--config", 'configs/dataset_comparison/aad_net.yaml',
        "--key", 'population',
        "--dataset", dataset,
        "--project", 'dataset_comparison',
        "--cross_val",
        "--finetuned",
        "--wandb",
    ]  
    print(cmd)
    subprocess.run(cmd)

    cmd = [
        "py", "train_models.py", 
        "--config", 'configs/dataset_comparison/cnn.yaml',
        "--key", 'subj_specific',
        "--dataset", dataset,
        "--project", 'dataset_comparison',
        "--cross_val",
        "--wandb",
    ]  
    print(cmd)
    subprocess.run(cmd)

    cmd = [
        "py", "evaluate.py", 
        "--config", 'configs/dataset_comparison/cnn.yaml',
        "--key", 'subj_specific',
        "--dataset", dataset,
        "--project", 'dataset_comparison',
        "--cross_val",
        "--wandb",
    ]  
    print(cmd)
    subprocess.run(cmd)

    # cmd = [
    #     "py", "train_linear.py", 
    #     "--config", 'configs/dataset_comparison/linear_models.yaml',
    #     "--key", 'subj_specific',
    #     "--dataset", dataset,
    #     "--project", 'dataset_comparison',
    #     "--cross_val",
    # ]  
    # print(cmd)
    # subprocess.run(cmd)

    # cmd = [
    #     "py", "eval_linear.py", 
    #     "--config", 'configs/dataset_comparison/linear_models.yaml',
    #     "--key", 'subj_specific',
    #     "--dataset", dataset,
    #     "--project", 'dataset_comparison',
    #     "--cross_val",
    #     "--wandb",
    # ]  
    # print(cmd)
    # subprocess.run(cmd)