"""
测试 DenseGNN + JARVIS mbj_bandgap 设置
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 减少 TensorFlow 日志

print("=" * 60)
print("测试 DenseGNN + JARVIS mbj_bandgap 设置")
print("=" * 60)

# 1. 测试数据集导入
print("\n[1/3] 测试数据集...")
try:
    from kgcnn.data.datasets.JarvisMbjBandgapDataset import JarvisMbjBandgapDataset
    print("    ✓ JarvisMbjBandgapDataset 导入成功")
except Exception as e:
    print(f"    ✗ 数据集导入失败: {e}")
    exit(1)

# 2. 测试模型导入
print("\n[2/3] 测试模型...")
try:
    from kgcnn.literature.DenseGNN import make_model_asu
    print("    ✓ DenseGNN make_model_asu 导入成功")
except Exception as e:
    print(f"    ✗ 模型导入失败: {e}")
    exit(1)

# 3. 测试配置文件
print("\n[3/3] 测试配置文件...")
try:
    from kgcnn.training.hyper import HyperParameter
    hyper = HyperParameter(
        hyper_info="training/hyper/hyper_jarvis_mbj_bandgap.py",
        hyper_category="DenseGNN"
    )
    hyper.verify()
    print("    ✓ 配置文件验证成功")
    print(f"    - 数据集: {hyper['data']['dataset']['class_name']}")
    print(f"    - 模型: {hyper['model']['class_name']}")
    print(f"    - 批次大小: {hyper['training']['fit']['batch_size']}")
    print(f"    - 训练轮数: {hyper['training']['fit']['epochs']}")
except Exception as e:
    print(f"    ✗ 配置文件测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✓ 所有测试通过！准备开始训练。")
print("=" * 60)
