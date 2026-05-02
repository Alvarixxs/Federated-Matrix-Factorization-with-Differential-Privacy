#!/bin/bash

set -e

SEEDS=(0 1 2 3 4)
SAMPLE_RATE=0.1
CLIP_NORM=1.0

run_experiment() {
    local DATASET=$1
    local K=$2
    local LR=$3
    local REG=$4
    local BATCH_SIZE=$5
    local ROUNDS=$6
    local LOCAL_EPOCHS=$7
    local DELTA=$8
    local N_CLIENTS=$9
    local NOISE_MULT=${10}

    for SEED in "${SEEDS[@]}"; do
        echo "    Seed $SEED..."
        python -m src.training.train_FL_DP \
            --dataset "$DATASET" --seed "$SEED" \
            --k "$K" --lr "$LR" --reg "$REG" \
            --batch_size "$BATCH_SIZE" --rounds "$ROUNDS" \
            --sample_rate "$SAMPLE_RATE" --clip_norm "$CLIP_NORM" \
            --local_epochs "$LOCAL_EPOCHS" \
            --noise_multiplier "$NOISE_MULT" \
            --n_clients "$N_CLIENTS"

        python -m src.evaluation.evaluate_model \
            --dataset "$DATASET" --seed "$SEED" \
            --model FL_DP \
            --k "$K" --lr "$LR" --reg "$REG" \
            --batch_size "$BATCH_SIZE" --rounds "$ROUNDS" \
            --local_epochs "$LOCAL_EPOCHS" \
            --noise_multiplier "$NOISE_MULT" \
            --n_clients "$N_CLIENTS" \
            --delta "$DELTA"
    done
}

# ml-100k
echo "Experimentos DP con distintos |C| en ml-100k..."

for N_CLIENTS in 50 200 500 943; do
    NOISE_MULTS=$(python3 -c "
nc = $N_CLIENTS
q = 0.1
C = 1.0
sigma_abs_list = [0.0075, 0.0125, 0.0175, 0.0225, 0.0275]
mults = [round(s * nc * q / (2 * C), 6) for s in sigma_abs_list]
print(' '.join(map(str, mults)))
")
    for NOISE_MULT in $NOISE_MULTS; do
        echo "  |C|=$N_CLIENTS, noise_mult=$NOISE_MULT..."
        run_experiment "ml-100k" 32 0.05 0.001 64 80 16 1e-3 "$N_CLIENTS" "$NOISE_MULT"
    done
done

# ml-1m
echo "Experimentos DP con distintos |C| en ml-1m..."

for N_CLIENTS in 500 1000 3000 6040; do
    NOISE_MULTS=$(python3 -c "
nc = $N_CLIENTS
q = 0.1
C = 1.0
sigma_abs_list = [0.003, 0.004, 0.006, 0.008, 0.009, 0.011, 0.012, 0.013, 0.014, 0.015]
mults = [round(s * nc * q / (2 * C), 6) for s in sigma_abs_list]
print(' '.join(map(str, mults)))
")
    for NOISE_MULT in $NOISE_MULTS; do
        echo "  |C|=$N_CLIENTS, noise_mult=$NOISE_MULT..."
        run_experiment "ml-1m" 64 0.1 0.001 64 100 16 1e-4 "$N_CLIENTS" "$NOISE_MULT"
    done
done

echo "Experimentos completados."