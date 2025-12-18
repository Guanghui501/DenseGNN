#!/bin/bash
# Training script for DenseGNN on JARVIS mbj_bandgap dataset

echo "========================================="
echo "DenseGNN Training on JARVIS mbj_bandgap"
echo "========================================="
echo ""

# 设置Python环境
export TF_CPP_MIN_LOG_LEVEL=2  # 减少TensorFlow日志输出

# 切换到项目目录
cd /home/user/DenseGNN

echo "配置信息:"
echo "- 模型: DenseGNN"
echo "- 数据集: JarvisMbjBandgapDataset"
echo "- 超参数文件: training/hyper/hyper_jarvis_mbj_bandgap.py"
echo "- 训练脚本: training/train_crystal.py"
echo ""

# 开始训练
echo "开始训练..."
python training/train_crystal.py \
    --hyper training/hyper/hyper_jarvis_mbj_bandgap.py \
    --category DenseGNN.make_model_asu \
    --gpu 0 \
    --seed 42

echo ""
echo "训练完成！"
echo "结果保存在 training/results/ 目录下"
