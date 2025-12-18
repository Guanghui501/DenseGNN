#!/usr/bin/env python
"""
DenseGNN 安装验证脚本
检查所有必要的依赖是否正确安装
"""

import sys

def check_installation():
    """检查所有核心依赖"""
    print("=" * 60)
    print("DenseGNN 安装验证")
    print("=" * 60)

    errors = []

    # 检查 TensorFlow
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  └─ 检测到 {len(gpus)} 个 GPU 设备")
        else:
            print(f"  └─ 未检测到 GPU (将使用 CPU)")
    except ImportError as e:
        errors.append(f"✗ TensorFlow 导入失败: {e}")
        print(errors[-1])

    # 检查化学信息学库
    try:
        import rdkit
        from rdkit import __version__
        print(f"✓ RDKit {__version__}")
    except ImportError as e:
        errors.append(f"✗ RDKit 导入失败: {e}")
        print(errors[-1])

    # 检查材料科学库
    try:
        import pymatgen
        try:
            version = pymatgen.__version__
        except AttributeError:
            version = "installed"
        print(f"✓ Pymatgen {version}")
    except ImportError as e:
        errors.append(f"✗ Pymatgen 导入失败: {e}")
        print(errors[-1])

    # 检查其他核心库
    packages = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('pandas', 'Pandas'),
        ('networkx', 'NetworkX'),
        ('ase', 'ASE'),
        ('matplotlib', 'Matplotlib'),
    ]

    for module_name, display_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name} {version}")
        except ImportError as e:
            errors.append(f"✗ {display_name} 导入失败: {e}")
            print(errors[-1])

    # 检查 DenseGNN 模块
    try:
        import kgcnn.literature.DenseGNN
        print("✓ DenseGNN 模块")
    except ImportError as e:
        errors.append(f"✗ DenseGNN 模块导入失败: {e}")
        print(errors[-1])

    # 检查其他 kgcnn 模块
    try:
        from kgcnn.literature.Schnet import make_model
        print("✓ kgcnn.literature.Schnet")
    except ImportError as e:
        print(f"⚠ kgcnn.literature.Schnet 导入失败 (非关键): {e}")

    try:
        from kgcnn.data.base import MemoryGraphDataset
        print("✓ kgcnn.data.base")
    except ImportError as e:
        print(f"⚠ kgcnn.data.base 导入失败 (非关键): {e}")

    print("=" * 60)

    if errors:
        print(f"\n发现 {len(errors)} 个错误。请检查上述错误信息。")
        return False
    else:
        print("\n✓ 所有核心依赖已成功安装！")
        print("\n您现在可以开始使用 DenseGNN 了。")
        print("参考 README.md 和 training/ 目录获取使用示例。")
        return True

if __name__ == "__main__":
    success = check_installation()
    sys.exit(0 if success else 1)
