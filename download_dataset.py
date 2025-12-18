#!/usr/bin/env python
"""
下载 JARVIS mbj_bandgap 数据集
"""
import os
import sys

# 设置 PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 70)
print("下载 JARVIS mbj_bandgap 数据集")
print("=" * 70)

try:
    from kgcnn.data.datasets.JarvisMbjBandgapDataset import JarvisMbjBandgapDataset

    print("\n正在初始化数据集...")
    print("首次运行将自动下载数据...")

    # 初始化数据集（会自动下载）
    dataset = JarvisMbjBandgapDataset(reload=False, verbose=10)

    print(f"\n✓ 数据集下载成功！")
    print(f"  - 数据集大小: {len(dataset)} 个样本")
    print(f"  - 标签名称: {dataset.label_names}")
    print(f"  - 标签单位: {dataset.label_units}")

    print("\n" + "=" * 70)
    print("数据集准备完成！现在可以开始训练。")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ 数据集下载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
