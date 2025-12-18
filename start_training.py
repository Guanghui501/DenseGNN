#!/usr/bin/env python
"""
启动 DenseGNN 在 JARVIS mbj_bandgap 数据集上的训练
此脚本确保正确的模块导入路径
"""
import sys
import os

# 将当前项目目录添加到 Python 路径的最前面
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# 设置环境变量以减少日志输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 70)
print("DenseGNN 训练 - JARVIS mbj_bandgap 数据集")
print("=" * 70)
print(f"项目目录: {project_dir}")
print(f"Python路径已设置: {sys.path[0]}")
print("=" * 70)

# 现在运行训练脚本
import subprocess
result = subprocess.run([
    sys.executable,
    os.path.join(project_dir, "training/train_crystal.py"),
    "--hyper", "training/hyper/hyper_jarvis_mbj_bandgap.py",
    "--category", "DenseGNN",
    "--seed", "42"
], env={**os.environ, 'PYTHONPATH': project_dir})

sys.exit(result.returncode)
