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
SAMPLE_RATE=0.1
CLIP_NORM=1.0
NOISE_MULTIPLIERS=(0.5 1.0 1.2 2.0)

echo "Iniciando entrenamiento en $DATASET..."

for SEED in "${SEEDS[@]}"; do
    echo "Seed $SEED..."

    echo "  Entrenando MF..."
    python -m src.training.train_MF \
        --dataset "$DATASET" --seed "$SEED" \
        --k "$K" --lr "$LR" \
        --batch_size "$BATCH_SIZE" --reg "$REG" --rounds "$ROUNDS"

    echo "  Entrenando FL..."
    python -m src.training.train_FL \
        --dataset "$DATASET" --seed "$SEED" \
        --k "$K" --lr "$LR" \
        --local_epochs "$LOCAL_EPOCHS" \
        --batch_size "$BATCH_SIZE" --reg "$REG" --rounds "$ROUNDS" \
        --sample_rate "$SAMPLE_RATE"

    for NOISE in "${NOISE_MULTIPLIERS[@]}"; do
        echo "  Entrenando FL_DP con noise_multiplier=$NOISE..."
        python -m src.training.train_FL_DP \
            --dataset "$DATASET" --seed "$SEED" \
            --k "$K" --lr "$LR" \
            --local_epochs "$LOCAL_EPOCHS" \
            --batch_size "$BATCH_SIZE" --reg "$REG" --rounds "$ROUNDS" \
            --sample_rate "$SAMPLE_RATE" --clip_norm "$CLIP_NORM" \
            --noise_multiplier "$NOISE"
    done
done

echo "Entrenamiento completado."