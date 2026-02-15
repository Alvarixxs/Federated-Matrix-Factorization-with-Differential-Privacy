#!/bin/bash

set -e  # Si algo falla, se detiene todo

# ==========================================================
# 🔧 Valores por defecto (idénticos a los otros scripts)
# ==========================================================

DATASET="ml-latest-small"  # Valor por defecto para dataset
SEED=0

K=32
LR=0.05
LOCAL_EPOCHS=4
BATCH_SIZE=64
REG=1e-3
ROUNDS=30

SAMPLE_RATE=0.1
CLIP_NORM=1.0
DELTA=1e-5

NOISE_MULTIPLIERS=(0.5 1 1.2 1.5 2 2.5 3 4)

# ==========================================================
# 🧠 Parseo de argumentos opcionales
# ==========================================================

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --k) K="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --local_epochs) LOCAL_EPOCHS="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --reg) REG="$2"; shift ;;
        --rounds) ROUNDS="$2"; shift ;;
        --delta) DELTA="$2"; shift ;;
        --noise_multipliers)
            shift
            NOISE_MULTIPLIERS=()
            while [[ "$1" != "" && "$1" != --* ]]; do
                NOISE_MULTIPLIERS+=("$1")
                shift
            done
            continue
            ;;
        *) echo "Argumento desconocido: $1"; exit 1 ;;
    esac
    shift
done

chmod +x ./scripts/run_splits.sh
chmod +x ./scripts/run_training.sh
chmod +x ./scripts/run_evaluations.sh
chmod +x ./scripts/run_mia.sh

echo "======================================="
echo "🚀 EJECUTANDO PIPELINE COMPLETO"
echo "======================================="

# ==========================================================
# 0️⃣ Crear splits para MIA
# ==========================================================

echo ""
echo "🔵 Ejecutando run_splits.sh..."

./scripts/run_splits.sh \
    --dataset "$DATASET" \
    --train_ratio 0.7 \
    --test_ratio 0.15 \
    --seed "$SEED"


# ==========================================================
# 1️⃣ Ejecutar entrenamiento
# ==========================================================

echo ""
echo "🔵 Ejecutando run_training.sh..."

./scripts/run_training.sh \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --k "$K" \
    --lr "$LR" \
    --local_epochs "$LOCAL_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --reg "$REG" \
    --rounds "$ROUNDS" \
    --sample_rate "$SAMPLE_RATE" \
    --clip_norm "$CLIP_NORM" \
    --noise_multipliers "${NOISE_MULTIPLIERS[@]}"

# ==========================================================
# 2️⃣ Ejecutar evaluación
# ==========================================================

echo ""
echo "📊 Ejecutando run_evaluations.sh..."

./scripts/run_evaluations.sh \
    --dataset "$DATASET" \
    --k "$K" \
    --lr "$LR" \
    --local_epochs "$LOCAL_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --reg "$REG" \
    --rounds "$ROUNDS" \
    --delta "$DELTA" \
    --noise_multipliers "${NOISE_MULTIPLIERS[@]}"

# ==========================================================
# 3️⃣ Ejecutar ataque de Membership Inference
# ==========================================================

echo ""
echo "🕵️‍♂️ Ejecutando run_mia.sh..."

./scripts/run_mia.sh \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --k "$K" \
    --lr "$LR" \
    --local_epochs "$LOCAL_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --reg "$REG" \
    --rounds "$ROUNDS" \
    --noise_multipliers "${NOISE_MULTIPLIERS[@]}"

echo ""
echo "======================================="
echo "🎉 PIPELINE COMPLETO FINALIZADO"
echo "======================================="
