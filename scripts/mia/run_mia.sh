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
NOISE_MULTIPLIERS=(0.5 1.0 1.2 2.0)

run_mia() {
    local MODEL=$1
    local SEED=$2
    local NOISE=$3

    ARGS="--dataset $DATASET --model $MODEL --k $K --lr $LR \
          --local_epochs $LOCAL_EPOCHS --batch_size $BATCH_SIZE \
          --reg $REG --rounds $ROUNDS --seed $SEED"

    if [[ -n "$NOISE" ]]; then
        ARGS="$ARGS --noise_multiplier $NOISE"
    fi

    python -m src.evaluation.extract_scores $ARGS
    python -m src.attacks.build_mia_dataset $ARGS
    python -m src.attacks.mia_threshold $ARGS
    python -m src.attacks.mia_classifier $ARGS
}

echo "Iniciando pipeline MIA en $DATASET..."

for SEED in "${SEEDS[@]}"; do
    echo "Seed $SEED..."

    echo "  Ejecutando MIA para MF..."
    run_mia "MF" "$SEED"

    echo "  Ejecutando MIA para FL..."
    run_mia "FL" "$SEED"

    for NOISE in "${NOISE_MULTIPLIERS[@]}"; do
        echo "  Ejecutando MIA para FL_DP con noise_multiplier=$NOISE..."
        run_mia "FL_DP" "$SEED" "$NOISE"
    done
done

echo "Pipeline MIA completado."