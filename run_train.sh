#!/bin/bash
#SBATCH --job-name=run_all
<<<<<<< HEAD
#SBATCH -t 02:40:00                  # estimated time # TODO: adapt to your needs
=======
#SBATCH -t 00:120:00                  # estimated time # TODO: adapt to your needs
>>>>>>> georg
#SBATCH -p grete:shared              # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                    # take 1 GPU, see https://docs.hpc.gwdg.de/compute_partitions/gpu_partitions/index.html for more options
#SBATCH --mem-per-gpu=8G             # setting the right constraints for the splitted gpu partitions
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=8            # number cores per task
#SBATCH --mail-user=h.siddiqui@stud.uni-goettingen.de                # send mail when job begins and ends
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate dnlp # Or whatever you called your environment.

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env 2> /dev/null

# Print out some git info.
module load git
echo -e "\nCurrent Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Latest Commit: $(git rev-parse --short HEAD)"
echo -e "Uncommitted Changes: $(git status --porcelain | wc -l)\n"

# Run the script:
# SST
#echo -e "\nStarting SST\n"
#python -u multitask_classifier.py --use_gpu --option finetune --task sst --hidden_dropout_prob 0.20 --epochs 10 --batch_size 64 --lr 1e-5

# STS
echo -e "\nStarting STS\n"
### Fine-Tune + Pretrained SimCSE###
# Standard
#python multitask_classifier.py --task sts --option finetune --regressor_type simple --forward_type raw_cls --sts_training_type simcse --epochs 5 --lr 3e-5 --batch_size 64 --use_gpu --use_pretrained_simcse --simcse_model_path "models/simcse_supervised/best_model_epoch3_corr0.8216.pt" --max_batches 10

# Sbert
#python multitask_classifier.py --task sts --option finetune --regressor_type sbert --forward_type sbert_mean --sts_training_type sbert --use_pretrained_simcse --simcse_model_path "models/simcse_supervised/best_model_epoch3_corr0.8216.pt" --max_batches 20 --epochs 3 --lr 1e-5 --batch_size 32 --use_gpu

# SimCSE
#python multitask_classifier.py --task sts --option finetune --regressor_type simple --forward_type raw_cls --sts_training_type simcse --use_pretrained_simcse --simcse_model_path "models/simcse_supervised/best_model_epoch3_corr0.8216.pt" --max_batches 15 --epochs 10 --lr 3e-5 --batch_size 64 --use_gpu

# SBert + SimCSE 
python multitask_classifier.py --task sts --option finetune --regressor_type simple --sts_training_type simcse_sbert --forward_type simcse_sbert --use_pretrained_simcse --simcse_model_path "models/simcse_supervised/best_model_epoch3_corr0.8216.pt" --max_batches 10 --epochs 7 --batch_size 64 --lr 2e-5 --alpha 0.975 --use_gpu

### Fine-Tune ###
# Standard
#python multitask_classifier.py --task sts --option finetune --regressor_type simple --forward_type pooler --sts_training_type standard --epochs 5 --lr 3e-5 --batch_size 64 --use_gpu 

# Sbert
#python multitask_classifier.py --task sts --option finetune --regressor_type sbert --forward_type sbert_mean --sts_training_type sbert --epochs 3 --lr 1e-5 --batch_size 32 --use_gpu 

# SimCSE
#python multitask_classifier.py --task sts --option finetune --regressor_type simple --forward_type raw_cls --sts_training_type simcse --epochs 10 --lr 3e-5 --batch_size 64 --use_gpu 

# SBert + SimCSE
#python multitask_classifier.py --task sts --option finetune --regressor_type simple --sts_training_type simcse_sbert --forward_type simcse_sbert --epochs 7 --batch_size 64 --lr 2e-5 --alpha 0.975 --use_gpu


### Pre Fine-Tune with SimCSE ###
#python simcse_pretrain.py --supervised --epochs 5 --batch_size 64 --lr 3e-5 --pooling_method mean --gradient_accumulation 2

# QQP
echo -e "\nStarting QQP\n"
# normal model
#python multitask_classifier.py --use_gpu --option finetune --task qqp --regressor_type qqp --hidden_dropout_prob 0.1 --epochs=6 
# new model
python multitask_classifier.py --use_gpu --option finetune --task qqp --regressor_type qqp --hidden_dropout_prob 0.1 --epochs=6 --use_pretrained_simcse 

# PTD-Bert
echo -e "\nStarting PTD-Bert\n"
#python -u multitask_classifier.py --use_gpu --option finetune --task etpc --hidden_dropout_prob 0.25 --epochs=20 --lr 1e-6 --batch_size 16

# Paraphrase Type Detection
echo -e "\nStarting PTD\n"
#python bart_detection.py --use_gpu

# Paraphrase Generation
echo -e "\nStarting PG\n"
#python -u bart_generation.py --use_gpu
