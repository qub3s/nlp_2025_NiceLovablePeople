#!/bin/bash
#SBATCH --job-name=sts_alpha_sweep_20seeds
#SBATCH -t 45:00:00  # Longer time for more seeds
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=4G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=./slurm_files/sts_test-%x-%j.out
#SBATCH --error=./slurm_files/sts_test-%x-%j.err

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate dnlp

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

# Define parameter ranges - focus on most promising values
ALPHAS=(0.99)
BATCH_SIZES_SBERT=(5 10 15 20 50 80 110 150 180)
BATCH_SIZES_SIMCSE=(5 10 15 20 30 50 70 90)
SEEDS=(11711 11712 11713 11714 11715 11716 11717 11718 11719 11720 
       11721 11722 11723 11724 11725)

# Create results directory
echo "Starting parameter sweeps with 25 seeds"

# Run simcse-sbert experiments
echo "Running simcse-sbert experiments"
for alpha in "${ALPHAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running simcse-sbert alpha=$alpha, seed=$seed"
        python multitask_classifier.py \
            --task sts \
            --option finetune \
            --sts_training_type simcse_sbert \
            --forward_type simcse_sbert \
            --use_pretrained_simcse \
            --simcse_model_path "models/simcse_supervised/best_model_epoch3_corr0.8216.pt" \
            --max_batches 10 \
            --epochs 5 \
            --batch_size 64 \
            --lr 2e-5 \
            --alpha "$alpha" \
            --seed "$seed" \
            --use_gpu \
            --save_results_only \
            --filepath "/dev/null"
        echo "Completed simcse-sbert alpha=$alpha, seed=$seed"
    done
done

# # Run SimCSE-only experiments
# echo "Running SimCSE-only experiments"
# for batch_size in "${BATCH_SIZES_SIMCSE[@]}"; do
#     for seed in "${SEEDS[@]}"; do
#         echo "Running SimCSE-only batch_size=$batch_size, seed=$seed"
#         python multitask_classifier.py \
#             --task sts \
#             --option finetune \
#             --sts_training_type simcse \
#             --forward_type raw_cls \
#             --use_pretrained_simcse \
#             --simcse_model_path "models/simcse_supervised/best_model_epoch3_corr0.8216.pt" \
#             --max_batches "$batch_size" \
#             --epochs 10 \
#             --batch_size 64 \
#             --lr 3e-5 \
#             --seed "$seed" \
#             --use_gpu \
#             --save_results_only \
#             --filepath "/dev/null"
#         echo "Completed SimCSE-only batch_size=$batch_size, seed=$seed"
#     done
# done

# # Run SBERT-only experiments
# echo "Running SBERT-only experiments"
# for batch_size in "${BATCH_SIZES_SBERT[@]}"; do
#     for seed in "${SEEDS[@]}"; do
#         echo "Running SBERT-only batch_size=$batch_size, seed=$seed"
#         python multitask_classifier.py \
#             --task sts \
#             --option finetune \
#             --sts_training_type sbert \
#             --forward_type sbert_mean \
#             --use_pretrained_simcse \
#             --simcse_model_path "models/simcse_supervised/best_model_epoch3_corr0.8216.pt" \
#             --max_batches "$batch_size" \
#             --epochs 3 \
#             --batch_size 32 \
#             --lr 1e-5 \
#             --seed "$seed" \
#             --use_gpu \
#             --save_results_only \
#             --filepath "/dev/null"
#         echo "Completed SBERT-only batch_size=$batch_size, seed=$seed"
#     done
# done


# Run simcse-sbert experiments
echo "Running simcse-sbert experiments"
for batch_size in "${BATCH_SIZES_SIMCSE[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running simcse-sbert alpha=$alpha, seed=$seed"
        python multitask_classifier.py \
            --task sts \
            --option finetune \
            --sts_training_type simcse_sbert \
            --forward_type simcse_sbert \
            --use_pretrained_simcse \
            --simcse_model_path "models/simcse_supervised/best_model_epoch3_corr0.8216.pt" \
            --max_batches 10 \
            --epochs 5 \
            --batch_size 64 \
            --lr 2e-5 \
            --alpha 0.975 \
            --seed "$seed" \
            --use_gpu \
            --save_results_only \
            --filepath "/dev/null"
        echo "Completed simcse-sbert alpha=$alpha, seed=$seed"
    done
done

echo "All parameter sweeps with 25 seeds completed!"