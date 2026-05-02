#!/bin/bash

set -e

SEEDS=(0 1 2 3 4)

# ml-100k
DATASET="ml-100k"
K=32
LR=0.05
REG=0.001
BATCH_SIZE=64
ROUNDS=80
LOCAL_EPOCHS=16
DELTA=1e-3
NOISE_MULTIPLIERS=(0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0)

echo "Evaluando FL_DP en $DATASET..."

for SEED in "${SEEDS[@]}"; do
    echo "Seed $SEED..."
    for NOISE in "${NOISE_MULTIPLIERS[@]}"; do
        echo "  noise_multiplier=$NOISE..."
        python -m src.evaluation.evaluate_model \
            --dataset "$DATASET" --seed "$SEED" \
            --model "FL_DP" \
            --k "$K" --lr "$LR" --reg "$REG" \
            --batch_size "$BATCH_SIZE" --rounds "$ROUNDS" \
            --local_epochs "$LOCAL_EPOCHS" \
            --noise_multiplier "$NOISE" \
            --delta "$DELTA"
    done
done

echo "Recopilando resultados FL_DP en $DATASET..."
python -m src.scripts.collect_dp_results \
    --dataset "$DATASET" \
    --k "$K" --lr "$LR" --reg "$REG" \
    --batch_size "$BATCH_SIZE" --rounds "$ROUNDS" \
    --local_epochs "$LOCAL_EPOCHS" \
    --noise_multipliers "${NOISE_MULTIPLIERS[@]}"

# ml-1m
DATASET="ml-1m"
K=64
LR=0.1
REG=0.001
BATCH_SIZE=64
ROUNDS=100
LOCAL_EPOCHS=16
DELTA=1e-4
NOISE_MULTIPLIERS=(0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0 8.0)

echo "Evaluando FL_DP en $DATASET..."

for SEED in "${SEEDS[@]}"; do
    echo "Seed $SEED..."
    for NOISE in "${NOISE_MULTIPLIERS[@]}"; do
        echo "  noise_multiplier=$NOISE..."
        python -m src.evaluation.evaluate_model \
            --dataset "$DATASET" --seed "$SEED" \
            --model "FL_DP" \
            --k "$K" --lr "$LR" --reg "$REG" \
            --batch_size "$BATCH_SIZE" --rounds "$ROUNDS" \
            --local_epochs "$LOCAL_EPOCHS" \
            --noise_multiplier "$NOISE" \
            --delta "$DELTA"
    done
done

echo "Recopilando resultados FL_DP en $DATASET..."
python -m src.scripts.collect_dp_results \
    --dataset "$DATASET" \
    --k "$K" --lr "$LR" --reg "$REG" \
    --batch_size "$BATCH_SIZE" --rounds "$ROUNDS" \
    --local_epochs "$LOCAL_EPOCHS" \
    --noise_multipliers "${NOISE_MULTIPLIERS[@]}"

echo "Evaluaciones FL_DP completadas."