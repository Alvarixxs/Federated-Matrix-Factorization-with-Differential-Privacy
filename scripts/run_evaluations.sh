#!/bin/bash

set -e

# ==========================================================
# ❗ Inicializar variables vacías
# ==========================================================

DATASET=""
K=""
LR=""
LOCAL_EPOCHS=""
BATCH_SIZE=""
REG=""
ROUNDS=""
DELTA=""
NOISE_MULTIPLIERS=()

# ==========================================================
# 🧠 Parseo de argumentos (todos obligatorios)
# ==========================================================

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --k) K="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --local_epochs) LOCAL_EPOCHS="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --reg) REG="$2"; shift ;;
        --rounds) ROUNDS="$2"; shift ;;
        --delta) DELTA="$2"; shift ;;
        --noise_multipliers)
            shift
            while [[ "$1" != "" && "$1" != --* ]]; do
                NOISE_MULTIPLIERS+=("$1")
                shift
            done
            continue
            ;;
        *) echo "❌ Argumento desconocido: $1"; exit 1 ;;
    esac
    shift
done

# ==========================================================
# 🔍 Verificación de argumentos obligatorios
# ==========================================================

if [[ -z "$DATASET" || -z "$K" || -z "$LR" || -z "$LOCAL_EPOCHS" || \
      -z "$BATCH_SIZE" || -z "$REG" || -z "$ROUNDS" || \
      -z "$DELTA" || \
      ${#NOISE_MULTIPLIERS[@]} -eq 0 ]]; then
    echo ""
    echo "❌ ERROR: Faltan argumentos obligatorios."
    echo ""
    echo "Uso correcto:"
    echo ""
    echo "./run_evaluations.sh \\"
    echo "  --dataset ml-latest-small \\"
    echo "  --k 32 \\"
    echo "  --lr 0.05 \\"
    echo "  --local_epochs 4 \\"
    echo "  --batch_size 64 \\"
    echo "  --reg 1e-3 \\"
    echo "  --rounds 30 \\"
    echo "  --delta 1e-5 \\"
    echo "  --noise_multipliers 0.5 1 2 4 8"
    echo ""
    exit 1
fi

echo "======================================="
echo "📊 INICIO DE EVALUACIONES"
echo "Dataset: $DATASET"
echo "======================================="

# ==========================================================
# 1️⃣ Evaluar MF
# ==========================================================

echo "🔵 Evaluando MF..."

python -m src.evaluation.evaluate_model \
    --dataset "$DATASET" \
    --model "MF" \
    --k "$K" \
    --lr "$LR" \
    --local_epochs 1 \
    --batch_size "$BATCH_SIZE" \
    --reg "$REG" \
    --rounds "$ROUNDS" \
    --delta "$DELTA"

# ==========================================================
# 2️⃣ Evaluar FL
# ==========================================================

echo "🟢 Evaluando FL..."

python -m src.evaluation.evaluate_model \
    --dataset "$DATASET" \
    --model "FL" \
    --k "$K" \
    --lr "$LR" \
    --local_epochs "$LOCAL_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --reg "$REG" \
    --rounds "$ROUNDS" \
    --delta "$DELTA"

# ==========================================================
# 3️⃣ Evaluar FL_DP
# ==========================================================

for NOISE in "${NOISE_MULTIPLIERS[@]}"
do
    echo "🟣 FL_DP | Noise Multiplier = $NOISE"

    python -m src.evaluation.evaluate_model \
        --dataset "$DATASET" \
        --model "FL_DP" \
        --k "$K" \
        --lr "$LR" \
        --local_epochs "$LOCAL_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --reg "$REG" \
        --rounds "$ROUNDS" \
        --noise_multiplier "$NOISE" \
        --delta "$DELTA"
done

echo "🎉 Evaluación completa finalizada."
