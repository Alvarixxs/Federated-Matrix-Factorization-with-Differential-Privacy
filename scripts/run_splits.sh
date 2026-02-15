#!/bin/bash

set -e

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --test_ratio) TEST_RATIO="$2"; shift ;;
        --train_ratio) TRAIN_RATIO="$2"; shift ;;
        *) echo "❌ Argumento desconocido: $1"; exit 1 ;;
    esac
    shift
done

# ==========================================================
# 🔍 Verificación de argumentos obligatorios
# ==========================================================

if [[ -z "$DATASET" || -z "$SEED" || -z "$TEST_RATIO" || -z "$TRAIN_RATIO" ]]; then
    echo ""
    echo "❌ ERROR: Faltan argumentos obligatorios."
    echo ""
    echo "Uso correcto:"
    echo ""
    echo "./run_splits.sh \\"
    echo "  --dataset ml-latest-small \\"
    echo "  --seed 0 \\"
    echo "  --test_ratio 0.15 \\"
    echo "  --train_ratio 0.7 \\"
    echo ""
    exit 1
fi

# =====================================
# EXECUTION
# =====================================

echo "======================================="
echo "Creating MIA splits"
echo "Dataset:      $DATASET"
echo "Train ratio:  $TRAIN_RATIO"
echo "Test ratio:   $TEST_RATIO"
echo "Seed:         $SEED"
echo "======================================="

python -m src.data.make_splits \
    --dataset $DATASET \
    --seed $SEED \
    --train_ratio $TRAIN_RATIO \
    --test_ratio $TEST_RATIO

echo ""
echo "✅ Splits creados exitosamente para el dataset '$DATASET' con seed $SEED."
