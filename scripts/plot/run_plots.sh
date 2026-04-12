#!/bin/bash

set -e

DATASET="ml-latest-small"
K=32
LR=0.05
LOCAL_EPOCHS=4
BATCH_SIZE=64
REG=0.001
ROUNDS=30
NOISE_MULTIPLIERS=(0.5 1.0 1.2 2.0)

NOISE_STR="${NOISE_MULTIPLIERS[*]}"

COMMON_ARGS="--dataset $DATASET \
    --k $K \
    --lr $LR \
    --local_epochs $LOCAL_EPOCHS \
    --batch_size $BATCH_SIZE \
    --reg $REG \
    --rounds $ROUNDS \
    --noise_multipliers $NOISE_STR"

echo "Generando plots RMSE para $DATASET..."

echo "RMSE Train vs Test..."
python -m src.plots.rmse_train_test $COMMON_ARGS

echo "RMSE vs Epsilon..."
python -m src.plots.rmse_eps $COMMON_ARGS

echo "Plots generados correctamente."