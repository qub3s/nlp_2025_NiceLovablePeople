#!/bin/bash
#SBATCH --job-name=sts_test_debug
#SBATCH -t 02:00:00
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

# Create output directories in advance
mkdir -p models/confidence_test

# DRAMATICALLY REDUCED PARAMETERS FOR TESTING - EVEN FURTHER
ALPHAS=(0.0)  # Only 1 alpha value for testing
BATCH_SIZES_SBERT=(5)  # Only 1 batch size
BATCH_SIZES_SIMCSE=(5) # Only 1 batch size  
SEEDS=(11711)       # Only 1 seed

# Modified functions with better error handling
run_simcse-sbert_experiment() {
    local alpha=$1
    local seed=$2
    
    echo "Running simcse-sbert alpha=$alpha, seed=$seed"
    
    OUTPUT_DIR="models/confidence_test/simcse_sbert_alpha_${alpha}_seed_${seed}"
    mkdir -p "$OUTPUT_DIR"
    
    # Test if directory was created successfully
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "ERROR: Could not create output directory $OUTPUT_DIR"
        return 1
    fi
    
    # REDUCED: max_batches=1, epochs=1, smaller batch size
    python multitask_classifier.py \
        --task sts \
        --option finetune \
        --regressor_type simple \
        --sts_training_type simcse_sbert \
        --forward_type simcse_sbert \
        --use_pretrained_simcse \
        --simcse_model_path "models/simcse_supervised/best_model_epoch3_corr0.8216.pt" \
        --max_batches 1 \
        --epochs 1 \
        --batch_size 8 \
        --lr 2e-5 \
        --alpha "$alpha" \
        --seed "$seed" \
        --use_gpu \
        --filepath "$OUTPUT_DIR"
    
    echo "Completed simcse-sbert alpha=$alpha, seed=$seed"
}

run_simcse_experiment() {
    local batch_size=$1
    local seed=$2
    
    echo "Running SimCSE-only batch_size=$batch_size, seed=$seed"
    
    OUTPUT_DIR="models/confidence_test/simcse_only_batch_${batch_size}_seed_${seed}"
    mkdir -p "$OUTPUT_DIR"
    
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "ERROR: Could not create output directory $OUTPUT_DIR"
        return 1
    fi
    
    # REDUCED: max_batches=1, epochs=1
    python multitask_classifier.py \
        --task sts \
        --option finetune \
        --regressor_type simple \
        --sts_training_type simcse \
        --forward_type raw_cls \
        --use_pretrained_simcse \
        --simcse_model_path "models/simcse_supervised/best_model_epoch3_corr0.8216.pt" \
        --max_batches 1 \
        --epochs 1 \
        --batch_size 8 \
        --lr 3e-5 \
        --seed "$seed" \
        --use_gpu \
        --filepath "$OUTPUT_DIR"
    
    echo "Completed SimCSE-only batch_size=$batch_size, seed=$seed"
}

run_sbert_experiment() {
    local batch_size=$1
    local seed=$2
    
    echo "Running SBERT-only batch_size=$batch_size, seed=$seed"
    
    OUTPUT_DIR="models/confidence_test/sbert_only_batch_${batch_size}_seed_${seed}"
    mkdir -p "$OUTPUT_DIR"
    
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "ERROR: Could not create output directory $OUTPUT_DIR"
        return 1
    fi
    
    # REDUCED: max_batches=1, epochs=1
    python multitask_classifier.py \
        --task sts \
        --option finetune \
        --regressor_type sbert \
        --forward_type sbert_mean \
        --sts_training_type sbert \
        --use_pretrained_simcse \
        --simcse_model_path "models/simcse_supervised/best_model_epoch3_corr0.8216.pt" \
        --max_batches 1 \
        --epochs 1 \
        --lr 1e-5 \
        --batch_size 8 \
        --use_gpu \
        --seed "$seed" \
        --filepath "$OUTPUT_DIR"
    
    echo "Completed SBERT-only batch_size=$batch_size, seed=$seed"
}

# Run sequentially for better debugging
echo "Running simcse-sbert experiments..."
for alpha in "${ALPHAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_simcse-sbert_experiment "$alpha" "$seed"
    done
done

echo "Running SimCSE-only experiments..."
for batch_size in "${BATCH_SIZES_SIMCSE[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_simcse_experiment "$batch_size" "$seed"
    done
done

echo "Running SBERT-only experiments..."
for batch_size in "${BATCH_SIZES_SBERT[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_sbert_experiment "$batch_size" "$seed"
    done
done

echo "Test completed!"