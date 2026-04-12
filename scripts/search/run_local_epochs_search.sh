#!/bin/bash

set -e

SEEDS=(0 1 2 3 4)
DELTA=1e-3
SAMPLE_RATE=0.1

LOCAL_EPOCHS_LIST=(1 2 4 8 16)

# ==================
# ml-100k
# ==================

DATASET="ml-100k"
K=32
LR=0.05
REG=0.001
BATCH_SIZE=64
ROUNDS=80

CONFIGS_100K=()

echo "Busqueda de local_epochs en $DATASET..."

for LOCAL_EPOCHS in "${LOCAL_EPOCHS_LIST[@]}"; do
    echo "local_epochs=$LOCAL_EPOCHS..."

    for SEED in "${SEEDS[@]}"; do
        echo "  Seed $SEED..."

        python -m src.training.train_FL \
            --dataset "$DATASET" --seed "$SEED" \
            --k "$K" --lr "$LR" --reg "$REG" \
            --batch_size "$BATCH_SIZE" --rounds "$ROUNDS" \
            --sample_rate "$SAMPLE_RATE" \
            --local_epochs "$LOCAL_EPOCHS"

        python -m src.evaluation.evaluate_model \
            --dataset "$DATASET" --seed "$SEED" \
            --model FL \
            --k "$K" --lr "$LR" --reg "$REG" \
            --batch_size "$BATCH_SIZE" --rounds "$ROUNDS" \
            --local_epochs "$LOCAL_EPOCHS" \
            --delta "$DELTA"
    done

    CONFIGS_100K+=("{\"k\":$K,\"lr\":$LR,\"reg\":$REG,\"batch_size\":$BATCH_SIZE,\"rounds\":$ROUNDS,\"local_epochs\":$LOCAL_EPOCHS}")
done

echo "Recopilando resultados $DATASET..."
python -m src.scripts.collect_hparam_results \
    --dataset "$DATASET" \
    --model FL \
    --n_seeds "${#SEEDS[@]}" \
    --configs "${CONFIGS_100K[@]}"

# ==================
# ml-1m
# ==================

DATASET="ml-1m"
K=64
LR=0.1
REG=0.001
BATCH_SIZE=64
ROUNDS=100

CONFIGS_1M=()

echo "Busqueda de local_epochs en $DATASET..."

for LOCAL_EPOCHS in "${LOCAL_EPOCHS_LIST[@]}"; do
    echo "local_epochs=$LOCAL_EPOCHS..."

    for SEED in "${SEEDS[@]}"; do
        echo "  Seed $SEED..."

        python -m src.training.train_FL \
            --dataset "$DATASET" --seed "$SEED" \
            --k "$K" --lr "$LR" --reg "$REG" \
            --batch_size "$BATCH_SIZE" --rounds "$ROUNDS" \
            --sample_rate "$SAMPLE_RATE" \
            --local_epochs "$LOCAL_EPOCHS"

        python -m src.evaluation.evaluate_model \
            --dataset "$DATASET" --seed "$SEED" \
            --model FL \
            --k "$K" --lr "$LR" --reg "$REG" \
            --batch_size "$BATCH_SIZE" --rounds "$ROUNDS" \
            --local_epochs "$LOCAL_EPOCHS" \
            --delta "$DELTA"
    done

    CONFIGS_1M+=("{\"k\":$K,\"lr\":$LR,\"reg\":$REG,\"batch_size\":$BATCH_SIZE,\"rounds\":$ROUNDS,\"local_epochs\":$LOCAL_EPOCHS}")
done

echo "Recopilando resultados $DATASET..."
python -m src.scripts.collect_hparam_results \
    --dataset "$DATASET" \
    --model FL \
    --n_seeds "${#SEEDS[@]}" \
    --configs "${CONFIGS_1M[@]}"

echo "Busqueda de local_epochs completada."