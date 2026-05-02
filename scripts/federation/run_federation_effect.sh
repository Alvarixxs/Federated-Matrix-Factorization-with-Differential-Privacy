#!/bin/bash

set -e

SEEDS=(0 1 2 3 4)
SAMPLE_RATE=0.1

# ml-100k
DATASET="ml-100k"
K=32
LR=0.05
REG=0.001
BATCH_SIZE=64
ROUNDS=80
LOCAL_EPOCHS=16
DELTA=1e-3
N_CLIENTS_LIST=(10 25 50 75 100 150 200 300 500 943)

echo "Efecto de la federacion en $DATASET..."

for N_CLIENTS in "${N_CLIENTS_LIST[@]}"; do
    echo "n_clients=$N_CLIENTS..."
    for SEED in "${SEEDS[@]}"; do
        echo "  Seed $SEED..."

        python -m src.training.train_FL \
            --dataset "$DATASET" --seed "$SEED" \
            --k "$K" --lr "$LR" --reg "$REG" \
            --batch_size "$BATCH_SIZE" --rounds "$ROUNDS" \
            --sample_rate "$SAMPLE_RATE" \
            --local_epochs "$LOCAL_EPOCHS" \
            --n_clients "$N_CLIENTS"

        python -m src.evaluation.evaluate_model \
            --dataset "$DATASET" --seed "$SEED" \
            --model FL \
            --k "$K" --lr "$LR" --reg "$REG" \
            --batch_size "$BATCH_SIZE" --rounds "$ROUNDS" \
            --local_epochs "$LOCAL_EPOCHS" \
            --n_clients "$N_CLIENTS" \
            --delta "$DELTA"
    done
done

echo "Recopilando resultados ml-100k..."
python -m src.scripts.collect_federation_results \
    --dataset "ml-100k" \
    --k 32 --lr 0.05 --reg 0.001 \
    --batch_size 64 --rounds 80 \
    --local_epochs 16 \
    --n_clients_list 10 25 50 75 100 150 200 300 500 943

# ml-1m
DATASET="ml-1m"
K=64
LR=0.1
REG=0.001
BATCH_SIZE=64
ROUNDS=100
LOCAL_EPOCHS=16
DELTA=1e-4
N_CLIENTS_LIST=(10 50 100 250 500 750 1000 1500 3000 6040)

echo "Efecto de la federacion en $DATASET..."

for N_CLIENTS in "${N_CLIENTS_LIST[@]}"; do
    echo "n_clients=$N_CLIENTS..."
    for SEED in "${SEEDS[@]}"; do
        echo "  Seed $SEED..."

        python -m src.training.train_FL \
            --dataset "$DATASET" --seed "$SEED" \
            --k "$K" --lr "$LR" --reg "$REG" \
            --batch_size "$BATCH_SIZE" --rounds "$ROUNDS" \
            --sample_rate "$SAMPLE_RATE" \
            --local_epochs "$LOCAL_EPOCHS" \
            --n_clients "$N_CLIENTS"

        python -m src.evaluation.evaluate_model \
            --dataset "$DATASET" --seed "$SEED" \
            --model FL \
            --k "$K" --lr "$LR" --reg "$REG" \
            --batch_size "$BATCH_SIZE" --rounds "$ROUNDS" \
            --local_epochs "$LOCAL_EPOCHS" \
            --n_clients "$N_CLIENTS" \
            --delta "$DELTA"
    done
done

echo "Recopilando resultados ml-1m..."
python -m src.scripts.collect_federation_results \
    --dataset "ml-1m" \
    --k 64 --lr 0.1 --reg 0.001 \
    --batch_size 64 --rounds 100 \
    --local_epochs 16 \
    --n_clients_list 10 50 100 250 500 750 1000 1500 3000 6040

echo "Experimentos del efecto de la federacion completados."