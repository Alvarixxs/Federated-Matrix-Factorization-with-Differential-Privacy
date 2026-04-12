#!/bin/bash

set -e

SEEDS=(0 1 2 3 4)
DELTA=1e-3

CONFIGS_SMALL=(
    '{"k":16,  "lr":0.05, "reg":0.01,   "batch_size":64, "rounds":80}'
    '{"k":32,  "lr":0.05, "reg":0.001,  "batch_size":64, "rounds":80}'
    '{"k":32,  "lr":0.05, "reg":0.001,  "batch_size":32, "rounds":80}'
    '{"k":32,  "lr":0.05, "reg":0.001,  "batch_size":64, "rounds":100}'
    '{"k":32,  "lr":0.1,  "reg":0.001,  "batch_size":64, "rounds":80}'
    '{"k":64,  "lr":0.05, "reg":0.001,  "batch_size":32, "rounds":80}'
    '{"k":64,  "lr":0.05, "reg":0.001,  "batch_size":64, "rounds":80}'
    '{"k":64,  "lr":0.05, "reg":0.001,  "batch_size":64, "rounds":100}'
    '{"k":64,  "lr":0.05, "reg":0.0001, "batch_size":32, "rounds":80}'
    '{"k":64,  "lr":0.01, "reg":0.001,  "batch_size":64, "rounds":80}'
    '{"k":128, "lr":0.05, "reg":0.001,  "batch_size":32, "rounds":80}'
    '{"k":128, "lr":0.05, "reg":0.001,  "batch_size":32, "rounds":100}'
    '{"k":128, "lr":0.05, "reg":0.001,  "batch_size":64, "rounds":80}'
    '{"k":32,  "lr":0.01, "reg":0.01,   "batch_size":32, "rounds":80}'
)

CONFIGS_1M=(
    '{"k":64,  "lr":0.05, "reg":0.001,  "batch_size":16,  "rounds":100}'
    '{"k":64,  "lr":0.05, "reg":0.001,  "batch_size":32,  "rounds":100}'
    '{"k":64,  "lr":0.05, "reg":0.001,  "batch_size":64,  "rounds":100}'
    '{"k":64,  "lr":0.05, "reg":0.001,  "batch_size":128, "rounds":100}'
    '{"k":64,  "lr":0.05, "reg":0.001,  "batch_size":64,  "rounds":150}'
    '{"k":64,  "lr":0.05, "reg":0.01,   "batch_size":16,  "rounds":100}'
    '{"k":64,  "lr":0.05, "reg":0.01,   "batch_size":64,  "rounds":100}'
    '{"k":64,  "lr":0.01, "reg":0.001,  "batch_size":32,  "rounds":100}'
    '{"k":64,  "lr":0.1,  "reg":0.001,  "batch_size":64,  "rounds":100}'
    '{"k":128, "lr":0.05, "reg":0.001,  "batch_size":16,  "rounds":100}'
    '{"k":128, "lr":0.05, "reg":0.001,  "batch_size":64,  "rounds":100}'
    '{"k":128, "lr":0.05, "reg":0.01,   "batch_size":32,  "rounds":100}'
    '{"k":128, "lr":0.05, "reg":0.001,  "batch_size":64,  "rounds":150}'
    '{"k":256, "lr":0.05, "reg":0.001,  "batch_size":64,  "rounds":100}'
    '{"k":256, "lr":0.05, "reg":0.01,   "batch_size":32,  "rounds":100}'
)

run_config() {
    local DATASET=$1
    local CONFIG=$2

    K=$(echo $CONFIG      | python3 -c "import sys,json; print(json.load(sys.stdin)['k'])")
    LR=$(echo $CONFIG     | python3 -c "import sys,json; print(json.load(sys.stdin)['lr'])")
    REG=$(echo $CONFIG    | python3 -c "import sys,json; print(json.load(sys.stdin)['reg'])")
    BS=$(echo $CONFIG     | python3 -c "import sys,json; print(json.load(sys.stdin)['batch_size'])")
    ROUNDS=$(echo $CONFIG | python3 -c "import sys,json; print(json.load(sys.stdin)['rounds'])")

    for SEED in "${SEEDS[@]}"; do
        echo "  Seed $SEED..."

        python -m src.training.train_MF \
            --dataset "$DATASET" --seed "$SEED" \
            --k "$K" --lr "$LR" --reg "$REG" \
            --batch_size "$BS" --rounds "$ROUNDS"

        python -m src.evaluation.evaluate_model \
            --dataset "$DATASET" --seed "$SEED" \
            --model MF \
            --k "$K" --lr "$LR" --reg "$REG" \
            --batch_size "$BS" --rounds "$ROUNDS" \
            --delta "$DELTA"
    done
}

echo "Busqueda de hiperparametros en ml-100k..."
for CONFIG in "${CONFIGS_SMALL[@]}"; do
    K=$(echo $CONFIG | python3 -c "import sys,json; print(json.load(sys.stdin)['k'])")
    LR=$(echo $CONFIG | python3 -c "import sys,json; print(json.load(sys.stdin)['lr'])")
    REG=$(echo $CONFIG | python3 -c "import sys,json; print(json.load(sys.stdin)['reg'])")
    echo "Configuracion: k=$K lr=$LR reg=$REG..."
    run_config "ml-100k" "$CONFIG"
done

echo "Recopilando resultados ml-100k..."
python -m src.scripts.collect_hparam_results \
    --dataset "ml-100k" \
    --model MF \
    --n_seeds "${#SEEDS[@]}" \
    --configs "${CONFIGS_SMALL[@]}"

echo "Busqueda de hiperparametros en ml-1m..."
for CONFIG in "${CONFIGS_1M[@]}"; do
    K=$(echo $CONFIG | python3 -c "import sys,json; print(json.load(sys.stdin)['k'])")
    LR=$(echo $CONFIG | python3 -c "import sys,json; print(json.load(sys.stdin)['lr'])")
    REG=$(echo $CONFIG | python3 -c "import sys,json; print(json.load(sys.stdin)['reg'])")
    echo "Configuracion: k=$K lr=$LR reg=$REG..."
    run_config "ml-1m" "$CONFIG"
done

echo "Recopilando resultados ml-1m..."
python -m src.scripts.collect_hparam_results \
    --dataset "ml-1m" \
    --model MF \
    --n_seeds "${#SEEDS[@]}" \
    --configs "${CONFIGS_1M[@]}"

echo "Busqueda de hiperparametros completada."