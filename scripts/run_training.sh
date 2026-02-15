#!/bin/bash

set -e

# ==========================================================
# 🧠 Parseo de argumentos (todos obligatorios)
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
        --sample_rate) SAMPLE_RATE="$2"; shift ;;
        --clip_norm) CLIP_NORM="$2"; shift ;;
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

if [[ -z "$DATASET" || -z "$SEED" || -z "$K" || -z "$LR" || \
      -z "$LOCAL_EPOCHS" || -z "$BATCH_SIZE" || -z "$REG" || \
      -z "$ROUNDS" || -z "$SAMPLE_RATE" || -z "$CLIP_NORM" || \
      ${#NOISE_MULTIPLIERS[@]} -eq 0 ]]; then
    echo ""
    echo "❌ ERROR: Faltan argumentos obligatorios."
    echo ""
    echo "Uso correcto:"
    echo ""
    echo "./run_training.sh \\"
    echo "  --dataset ml-latest-small \\"
    echo "  --seed 0 \\"
    echo "  --k 32 \\"
    echo "  --lr 0.05 \\"
    echo "  --local_epochs 4 \\"
    echo "  --batch_size 64 \\"
    echo "  --reg 1e-3 \\"
    echo "  --rounds 30 \\"
    echo "  --sample_rate 0.1 \\"
    echo "  --clip_norm 1.0 \\"
    echo "  --noise_multipliers 0.5 1 2 4 8"
    echo ""
    exit 1
fi

echo "======================================="
echo "🚀 INICIO DE EXPERIMENTOS"
echo "Dataset: $DATASET"
echo "======================================="

# ==========================================================
# 1️⃣ MF
# ==========================================================

echo "🔵 Entrenando MF..."

python -m src.training.train_MF \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --k "$K" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --reg "$REG" \
    --rounds "$ROUNDS"

echo "✅ MF completado."

# ==========================================================
# 2️⃣ FL
# ==========================================================

echo "🟢 Entrenando FL..."

python -m src.training.train_FL \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --k "$K" \
    --lr "$LR" \
    --local_epochs "$LOCAL_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --reg "$REG" \
    --rounds "$ROUNDS" \
    --sample_rate "$SAMPLE_RATE"

echo "✅ FL completado."

# ==========================================================
# 3️⃣ FL + DP
# ==========================================================

for NOISE in "${NOISE_MULTIPLIERS[@]}"
do
    echo "🟣 FL_DP | Noise Multiplier = $NOISE"

    python -m src.training.train_FL_DP \
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
        --noise_multiplier "$NOISE"

    echo "✅ FL_DP completado para noise_multiplier=$NOISE"
done

echo "🎉 Entrenamiento completo finalizado."
