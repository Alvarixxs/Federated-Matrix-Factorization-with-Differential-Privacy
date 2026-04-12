#!/bin/bash

set -e

DATASET="ml-latest-small"
SEEDS=(0 1 2 3 4)
K=32
LR=0.05
LOCAL_EPOCHS=4
BATCH_SIZE=64
REG=0.001
ROUNDS=30
DELTA=1e-5
NOISE_MULTIPLIERS=(0.5 1.0 1.2 2.0)

echo "Iniciando evaluaciones en $DATASET..."

for SEED in "${SEEDS[@]}"; do
    echo "Seed $SEED..."

    echo "  Evaluando MF..."
    python -m src.evaluation.evaluate_model \
        --dataset "$DATASET" --seed "$SEED" \
        --model "MF" \
        --k "$K" --lr "$LR" \
        --batch_size "$BATCH_SIZE" --reg "$REG" --rounds "$ROUNDS" \
        --delta "$DELTA"

    echo "  Evaluando FL..."
    python -m src.evaluation.evaluate_model \
        --dataset "$DATASET" --seed "$SEED" \
        --model "FL" \
        --k "$K" --lr "$LR" \
        --local_epochs "$LOCAL_EPOCHS" \
        --batch_size "$BATCH_SIZE" --reg "$REG" --rounds "$ROUNDS" \
        --delta "$DELTA"

    for NOISE in "${NOISE_MULTIPLIERS[@]}"; do
        echo "  Evaluando FL_DP con noise_multiplier=$NOISE..."
        python -m src.evaluation.evaluate_model \
            --dataset "$DATASET" --seed "$SEED" \
            --model "FL_DP" \
            --k "$K" --lr "$LR" \
            --local_epochs "$LOCAL_EPOCHS" \
            --batch_size "$BATCH_SIZE" --reg "$REG" --rounds "$ROUNDS" \
            --noise_multiplier "$NOISE" \
            --delta "$DELTA"
    done
done

echo "Evaluaciones completadas."