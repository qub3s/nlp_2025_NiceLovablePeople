#!/bin/bash
#SBATCH --job-name=sts_test_debug
#SBATCH -t 02:00:00  # Much shorter time
#SBATCH -p grete:shared
#SBATCH -G A100:1    # Only 1 GPU for testing
#SBATCH --mem-per-gpu=4G
#SBATCH --nodes=1
#SBATCH --ntasks=1   # Single task for debugging
#SBATCH --cpus-per-task=2
#SBATCH --output=./slurm_files/sts_test-%x-%j.out
#SBATCH --error=./slurm_files/sts_test-%x-%j.err

# ... rest of the header remains the same ...

# DRAMATICALLY REDUCED PARAMETERS FOR TESTING
ALPHAS=(0.0 0.5 1.0)  # Only 3 alpha values
BATCH_SIZES_SBERT=(5 20)  # Only 2 batch sizes
BATCH_SIZES_SIMCSE=(5 20) # Only 2 batch sizes  
SEEDS=(11711 11712)       # Only 2 seeds

# Also reduce training parameters in the functions:
run_simcse-sbert_experiment() {
    local alpha=$1
    local seed=$2
    
    echo "Running simcse-sbert alpha=$alpha, seed=$seed"
    
    OUTPUT_DIR="models/confidence_test/simcse_sbert_alpha_${alpha}_seed_${seed}"
    mkdir -p "$OUTPUT_DIR"
    
    # REDUCED: max_batches=2, epochs=2
    python multitask_classifier.py \
        --task sts \
        --option finetune \
        --regressor_type simple \
        --sts_training_type simcse_sbert \
        --forward_type simcse_sbert \
        --use_pretrained_simcse \
        --simcse_model_path "models/simcse_supervised/best_model_epoch3_corr0.8216.pt" \
        --max_batches 2 \          # REDUCED
        --epochs 2 \               # REDUCED
        --batch_size 16 \          # REDUCED
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
    
    # REDUCED: max_batches=2, epochs=2
    python multitask_classifier.py \
        --task sts \
        --option finetune \
        --regressor_type simple \
        --sts_training_type simcse \
        --forward_type raw_cls \
        --use_pretrained_simcse \
        --simcse_model_path "models/simcse_supervised/best_model_epoch3_corr0.8216.pt" \
        --max_batches 2 \          # REDUCED (use fixed small value)
        --epochs 2 \               # REDUCED
        --batch_size 16 \          # REDUCED
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
    
    # REDUCED: max_batches=2, epochs=1
    python multitask_classifier.py \
    --task sts \
    --option finetune \
    --regressor_type sbert \
    --forward_type sbert_mean \
    --sts_training_type sbert \
    --use_pretrained_simcse \
    --simcse_model_path "models/simcse_supervised/best_model_epoch3_corr0.8216.pt" \
    --max_batches 2 \          # REDUCED (use fixed small value)
    --epochs 1 \               # REDUCED
    --lr 1e-5 \
    --batch_size 16 \          # REDUCED
    --use_gpu \
    --seed "$seed" \
    --filepath "$OUTPUT_DIR"
    
    echo "Completed SBERT-only batch_size=$batch_size, seed=$seed"
}

export -f run_simcse-sbert_experiment
export -f run_simcse_experiment
export -f run_sbert_experiment

# Run experiments in parallel using GNU parallel if available
# Otherwise, use nested loops
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for parallel execution"
    
    # Run simcse-sbert experiments
    echo "Running simcse-sbert experiments..."
    parallel -j 4 run_simcse-sbert_experiment ::: "${ALPHAS[@]}" ::: "${SEEDS[@]}"
    
    # Run SimCSE-only experiments
    echo "Running SimCSE-only experiments..."
    parallel -j 4 run_simcse_experiment ::: "${BATCH_SIZES_SIMCSE[@]}" ::: "${SEEDS[@]}"
    
    # Run SBERT-only experiments
    echo "Running SBERT-only experiments..."
    parallel -j 4 run_sbert_experiment ::: "${BATCH_SIZES_SBERT[@]}" ::: "${SEEDS[@]}"
    
else
    echo "GNU parallel not found, running sequentially"
    
    # Run simcse-sbert experiments
    echo "Running simcse-sbert experiments..."
    for alpha in "${ALPHAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_simcse-sbert_experiment "$alpha" "$seed"
        done
    done
    
    # Run SimCSE-only experiments
    echo "Running SimCSE-only experiments..."
    for batch_size in "${BATCH_SIZES_SIMCSE[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_simcse_experiment "$batch_size" "$seed"
        done
    done
    
    # Run SBERT-only experiments
    echo "Running SBERT-only experiments..."
    for batch_size in "${BATCH_SIZES_SBERT[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_sbert_experiment "$batch_size" "$seed"
        done
    done
fi

echo "All parameter sweeps with 25 seeds completed!"