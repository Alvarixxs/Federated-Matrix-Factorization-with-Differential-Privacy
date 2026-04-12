#!/bin/bash

set -e

DATASET_SMALL="ml-100k"
DATASET_1M="ml-1m"
SEED=0
TRAIN_RATIO=0.7
VAL_RATIO=0.15
TEST_RATIO=0.15

echo "Creando splits para $DATASET_SMALL..."
python -m src.data.make_splits \
    --dataset "$DATASET_SMALL" \
    --seed "$SEED" \
    --train_ratio "$TRAIN_RATIO" \
    --val_ratio "$VAL_RATIO" \
    --test_ratio "$TEST_RATIO"

echo "Creando splits para $DATASET_1M..."
python -m src.data.make_splits \
    --dataset "$DATASET_1M" \
    --seed "$SEED" \
    --train_ratio "$TRAIN_RATIO" \
    --val_ratio "$VAL_RATIO" \
    --test_ratio "$TEST_RATIO"

echo "Splits creados correctamente."