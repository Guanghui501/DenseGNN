#!/bin/bash
# 简化的训练启动脚本

cd /home/user/DenseGNN

export PYTHONPATH=/home/user/DenseGNN:$PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=2

echo "=========================================="
echo "启动 DenseGNN 训练 - JARVIS mbj_bandgap"
echo "=========================================="

python training/train_crystal.py \
    --hyper training/hyper/hyper_jarvis_mbj_bandgap.py \
    --category DenseGNN \
    --make make_model_asu \
    --seed 42

echo ""
echo "训练完成！"
