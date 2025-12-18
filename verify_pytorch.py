#!/usr/bin/env python
"""验证 PyTorch backend 配置"""

import os
import sys

# 设置 PyTorch backend（必须在导入 keras 之前）
os.environ['KERAS_BACKEND'] = 'torch'

print("=" * 60)
print("PyTorch Backend 验证")
print("=" * 60)

# 1. 检查 PyTorch
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        print(f"  GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Compute capability: {props.major}.{props.minor}")
            print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("  ⚠ GPU 未检测到，将使用 CPU")

except ImportError as e:
    print(f"✗ PyTorch 未安装: {e}")
    print("\n安装命令:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

print()

# 2. 检查 Keras backend
try:
    import keras
    backend_name = keras.backend.backend()

    if backend_name == "torch":
        print(f"✓ Keras backend: {backend_name}")
    else:
        print(f"✗ Keras backend: {backend_name} (应该是 'torch')")
        print("\n请设置环境变量:")
        print("  export KERAS_BACKEND=torch")
        sys.exit(1)

except ImportError as e:
    print(f"✗ Keras 未安装: {e}")
    print("\n安装命令:")
    print("  pip install keras>=3.0.0")
    sys.exit(1)

print()

# 3. 检查 kgcnn
try:
    import kgcnn
    version = getattr(kgcnn, '__kgcnn_version__', 'unknown')
    print(f"✓ kgcnn version: {version}")

    # 尝试导入关键模块
    from kgcnn.literature.DenseGNN import make_model
    print("  ✓ DenseGNN model available")

    from kgcnn.data.datasets.MatProjectLogKVRHDataset import MatProjectLogKVRHDataset
    print("  ✓ MatProjectLogKVRHDataset available")

except ImportError as e:
    print(f"✗ kgcnn 导入失败: {e}")
    sys.exit(1)

print()

# 4. 测试简单的 PyTorch 操作
try:
    if torch.cuda.is_available():
        print("测试 GPU 操作...")
        device = torch.device("cuda:0")
        x = torch.randn(1000, 1000, device=device)
        y = torch.matmul(x, x)
        print(f"  ✓ GPU 计算成功 (设备: {y.device})")
    else:
        print("测试 CPU 操作...")
        x = torch.randn(100, 100)
        y = torch.matmul(x, x)
        print(f"  ✓ CPU 计算成功")

except Exception as e:
    print(f"  ✗ 计算测试失败: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("✓ 所有检查通过！可以开始训练")
print("=" * 60)
print("\n运行训练命令:")
print("  cd /home/user/DenseGNN")
print("  KERAS_BACKEND=torch python training/train_crystal.py --hyper training/hyper/hyper_mp_log_kvrh.py")
print()
