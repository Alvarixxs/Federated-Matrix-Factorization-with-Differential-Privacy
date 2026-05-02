#!/bin/bash

set -e

SEEDS=(0 1 2 3 4)
SAMPLE_RATE=0.1
CLIP_NORM=1.0

# ml-100k
DATASET="ml-100k"
K=32
LR=0.05
REG=0.001
BATCH_SIZE=64
ROUNDS=80
LOCAL_EPOCHS=16
NOISE_MULTIPLIERS=(0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0)

echo "Entrenando FL_DP en $DATASET..."

for SEED in "${SEEDS[@]}"; do
    echo "Seed $SEED..."
    for NOISE in "${NOISE_MULTIPLIERS[@]}"; do
        echo "  noise_multiplier=$NOISE..."
        python -m src.training.train_FL_DP \
            --dataset "$DATASET" --seed "$SEED" \
            --k "$K" --lr "$LR" --reg "$REG" \
            --batch_size "$BATCH_SIZE" --rounds "$ROUNDS" \
            --sample_rate "$SAMPLE_RATE" --clip_norm "$CLIP_NORM" \
            --local_epochs "$LOCAL_EPOCHS" \
            --noise_multiplier "$NOISE"
    done
done

echo "Entrenamiento FL_DP completado en $DATASET."

echo "Recopilando resultados FL_DP en ml-100k..."
python -m src.scripts.collect_dp_results \
    --dataset "ml-100k" \
    --k 32 --lr 0.05 --reg 0.001 \
    --batch_size 64 --rounds 80 \
    --local_epochs 16 \
    --noise_multipliers 0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0

# ml-1m
DATASET="ml-1m"
K=64
LR=0.1
REG=0.001
BATCH_SIZE=64
ROUNDS=100
LOCAL_EPOCHS=16
NOISE_MULTIPLIERS=(0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0 8.0)

echo "Entrenando FL_DP en $DATASET..."

for SEED in "${SEEDS[@]}"; do
    echo "Seed $SEED..."
    for NOISE in "${NOISE_MULTIPLIERS[@]}"; do
        echo "  noise_multiplier=$NOISE..."
        python -m src.training.train_FL_DP \
            --dataset "$DATASET" --seed "$SEED" \
            --k "$K" --lr "$LR" --reg "$REG" \
            --batch_size "$BATCH_SIZE" --rounds "$ROUNDS" \
            --sample_rate "$SAMPLE_RATE" --clip_norm "$CLIP_NORM" \
            --local_epochs "$LOCAL_EPOCHS" \
            --noise_multiplier "$NOISE"
    done
done

echo "Entrenamiento FL_DP completado en $DATASET."

echo "Recopilando resultados FL_DP en ml-1m..."
python -m src.scripts.collect_dp_results \
    --dataset "ml-1m" \
    --k 64 --lr 0.1 --reg 0.001 \
    --batch_size 64 --rounds 100 \
    --local_epochs 16 \
    --noise_multipliers 0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0 8.0