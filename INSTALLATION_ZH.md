# DenseGNN 安装指南 (中文)

## 安装状态

✓ **已成功安装**

## 已安装的核心组件

- **TensorFlow 2.17.1** - 深度学习框架
- **RDKit** - 化学信息学工具
- **Pymatgen** - 材料科学库
- **NumPy 1.26.4** - 数值计算
- **SciPy, Pandas, Matplotlib** - 科学计算和可视化
- **NetworkX** - 图处理
- **ASE** - 原子模拟环境
- **DenseGNN** - 本项目模块

## 快速验证

运行以下命令验证安装：

```bash
python verify_installation.py
```

或者手动测试：

```python
# 测试 TensorFlow
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")

# 测试 DenseGNN
import kgcnn.literature.DenseGNN
print("DenseGNN 模块导入成功")
```

## 系统信息

- **Python 版本**: 3.11.14
- **GPU 支持**: 当前未检测到 CUDA 驱动（使用 CPU）

### 启用 GPU 支持（可选）

如果您有 NVIDIA GPU 并希望加速训练，需要：

1. 安装 NVIDIA GPU 驱动
2. 安装 CUDA Toolkit (建议 11.8 或 12.x)
3. 安装 cuDNN (对应 CUDA 版本)

参考：https://www.tensorflow.org/install/gpu

## 使用示例

### 1. 查看可用模型

```python
from kgcnn.literature.DenseGNN import make_model_asu

# 查看 DenseGNN 模型
help(make_model_asu)
```

### 2. 运行训练示例

查看 `training/` 目录中的训练脚本和配置文件：

```bash
# 列出可用的训练配置
ls training/hyper/

# 查看训练脚本
ls training/*.py
```

### 3. 使用数据集

```python
from kgcnn.data.datasets.QM9Dataset import QM9Dataset

# 加载 QM9 数据集（自动下载）
dataset = QM9Dataset()
print(f"数据集大小: {len(dataset)}")
```

### 4. 构建模型

```python
import tensorflow as tf
from kgcnn.literature.DenseGNN import make_model_asu

# 创建 DenseGNN 模型
model = make_model_asu(
    name="DenseGNN",
    inputs=[
        {"shape": (None, 41), "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 11), "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}
    ],
    # 更多参数...
)

model.summary()
```

## 文档和资源

- **项目 README**: `README.md`
- **训练示例**: `training/` 目录
- **Jupyter 笔记本**: `notebooks/` 目录
- **API 文档**: `docs/` 目录
- **在线文档**: https://kgcnn.readthedocs.io/

## 常见数据集

DenseGNN 支持多个基准数据集，包括：

- **QM9** - 分子性质预测
- **ESOL, FreeSolv, Lipop** - 溶解度预测
- **Materials Project** - 晶体材料性质
- **JARVIS-DFT** - DFT 计算的材料数据
- **OC22** - 催化剂数据集
- **Matbench** - 材料基准测试

数据集会自动下载到 `~/.kgcnn/datasets/`

## 性能优化建议

1. **使用 GPU**: 如果有 NVIDIA GPU，安装 CUDA 支持可显著加速训练
2. **批处理大小**: 根据内存调整批处理大小
3. **数据预处理**: 预先计算并缓存图结构表示
4. **混合精度**: 使用 TensorFlow 的混合精度训练

## 故障排除

### 导入错误

如果遇到模块导入错误，尝试：

```bash
# 重新安装项目
pip install -e .

# 或者只安装特定依赖
pip install tensorflow rdkit pymatgen
```

### 版本冲突

当前已知的版本要求：
- TensorFlow 2.17.1 需要 NumPy < 2.0.0, >= 1.23.5
- Pymatgen 需要 NumPy >= 1.25.0

我们已经设置为 NumPy 1.26.4 来平衡这些要求。

### GPU 未检测到

如果安装了 CUDA 但 TensorFlow 未检测到：

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

确保 CUDA 和 TensorFlow 版本兼容。

## 引用

如果您在研究中使用 DenseGNN，请引用：

```bibtex
Du, H., Wang, J., Hui, J. et al. DenseGNN: universal and scalable deeper
graph neural networks for high-performance property prediction in crystals
and molecules. npj Comput Mater 10, 292 (2024).
https://doi.org/10.1038/s41524-024-01444-x
```

## 获取帮助

- **GitHub Issues**: https://github.com/aimat-lab/gcnn_keras/issues
- **文档**: https://kgcnn.readthedocs.io/
- **论文**: https://www.nature.com/articles/s41524-024-01444-x

## 下一步

1. 运行 `python verify_installation.py` 验证安装
2. 浏览 `notebooks/` 目录中的教程
3. 查看 `training/` 目录中的训练示例
4. 阅读论文了解 DenseGNN 的架构和原理

祝使用愉快！
