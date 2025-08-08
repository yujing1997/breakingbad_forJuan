import os
import argparse
import subprocess

def queue_job(account:str="def-senger", dataset_name:str="Dataset001_pet_zscore_noMD", 
              fold:int=0, dropout:bool=True, days:int=4):

    if dropout:
        dropout_prefix = ""
    else:
        dropout_prefix = "dropout_trans=0 "
    nnunet_std_prefix = "nnunet_std=1 "

    project_folder = "/home/sebquet/projects/rrg-senger-ab/sebquet/HECKTOR2025"
    job_name = f"ds_{dataset_name[7:]}_fold_{fold}"
    bash_string = f"""#!/bin/bash
#SBATCH --array=1-{days}%1   # jobnb is the number of jobs in the chain. This bash script will be executed 4 timeswiht 1 simultaneous job at a time.
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:a100:1
#SBATCH --account={account}
#SBATCH --mail-user=sebastien.quetin@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name={job_name}
#Environment setup
module restore hecktorModules
cd {project_folder}
source hecktorenv/bin/activate
export nnUNet_raw={project_folder}/nnUNet_raw
export nnUNet_results={project_folder}/nnUNet_results

cd $SLURM_TMPDIR
mkdir nnUNet_preprocessed
echo "Copying Dataset to local node scratch..." 
cp -r {project_folder}/nnUNet_preprocessed/{dataset_name} $SLURM_TMPDIR/nnUNet_preprocessed
export nnUNet_preprocessed=$SLURM_TMPDIR/nnUNet_preprocessed

# Training
echo "Starting the training"
export nnUNet_n_proc_DA=12
{nnunet_std_prefix}{dropout_prefix}nnUNetv2_train {dataset_name} 3d_fullres_bs8 {fold} --c

# The train command creates the splits.json file, which we want to save
cp $SLURM_TMPDIR/nnUNet_preprocessed/{dataset_name}/splits_final.json {project_folder}/nnUNet_preprocessed/{dataset_name}/splits_final.json

# Usage:
# sbatch training_narval.sh Dataset001_pet_zscore_noMD 0
"""
    bash_save_dir = os.path.join(project_folder, "nnUNet_results", dataset_name, f"nnUNetTrainer__nnUNetPlans__3d_fullres_bs8/fold_{fold}")
    os.makedirs(bash_save_dir, exist_ok=True)
    batch_filename = os.path.join(bash_save_dir, "bash_script.sh")
    with open(batch_filename, "w") as myfile:
        myfile.write(bash_string)

    
    # Run Slurm Batch Script
    command = "sbatch {}".format(batch_filename)
    print(command)
    subprocess.call(command, shell=True, cwd=bash_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Queue a job for training on Narval')
    parser.add_argument('-a', '--account', type=str, help='The account to use on Narval')
    parser.add_argument('-dn', '--dataset_name', type=str, help='The dataset name to use')
    parser.add_argument('-f', '--fold', type=int, help='The fold to train on')
    parser.add_argument('--no-dropout', action='store_true', help='Use dropout')
    parser.add_argument('--days', type=int, help='Number of days to train', default=4)
    args = parser.parse_args()
    queue_job(args.account, args.dataset_name, args.fold, not(args.no_dropout), args.days)