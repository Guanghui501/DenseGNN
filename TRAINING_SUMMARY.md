# DenseGNN 在 JARVIS mbj_bandgap 上的训练总结

## 环境配置完成

### ✅ 已完成的任务

1. **安装依赖包**
   - TensorFlow 2.17.1
   - RDKit, Pymatgen, NetworkX, ASE 等科学计算包
   - dm-tree 等必要模块

2. **修复代码兼容性**
   - 创建 `kgcnn/__init__.py` 文件
   - 添加 `__kgcnn_version__` 版本标识
   - 注释掉不兼容的 `tensorflow_addons` 导入
   - 复制 kgcnn.io 模块到本地项目

3. **创建训练配置**
   - 配置文件: `training/hyper/hyper_jarvis_mbj_bandgap.py`
   - 模型: DenseGNN (make_model_asu)
   - 数据集: JarvisMbjBandgapDataset
   - 训练参数:
     - 批次大小: 64
     - 训练轮数: 300
     - 5折交叉验证
     - 学习率调度器

4. **创建启动脚本**
   - `run_training_simple.sh` - 简化的训练启动脚本
   - `start_training.py` - Python 训练启动脚本
   - `test_setup.py` - 环境测试脚本

## 📋 待完成任务

### 数据集准备

JARVIS mbj_bandgap 数据集需要手动下载。数据集文件应放置在：
```
/home/datasets/jarvis_dft_3d_mbj_bandgap/mbj_bandgap.csv
```

#### 选项 1: 从 JARVIS 官方下载

1. 访问 JARVIS-DFT 数据库:
   - 网站: https://jarvis.nist.gov/
   - 或直接从: https://github.com/usnistgov/jarvis

2. 下载 mbj_bandgap 数据集（CSV 格式）

3. 创建目录并放置数据:
   ```bash
   mkdir -p /home/datasets/jarvis_dft_3d_mbj_bandgap
   cp mbj_bandgap.csv /home/datasets/jarvis_dft_3d_mbj_bandgap/
   ```

#### 选项 2: 使用已有 kgcnn 数据集

如果已经通过 kgcnn 下载过数据，数据可能在：
```bash
~/.kgcnn/datasets/jarvis_dft_3d_mbj_bandgap/
```

可以创建符号链接：
```bash
mkdir -p /home/datasets
ln -s ~/.kgcnn/datasets/jarvis_dft_3d_mbj_bandgap /home/datasets/
```

## 🚀 开始训练

数据集准备好后，运行：

```bash
cd /home/user/DenseGNN
./run_training_simple.sh
```

或者：

```bash
export PYTHONPATH=/home/user/DenseGNN:$PYTHONPATH
python training/train_crystal.py \
    --hyper training/hyper/hyper_jarvis_mbj_bandgap.py \
    --category DenseGNN \
    --make make_model_asu \
    --seed 42
```

## 📊 训练配置详情

### 模型配置 (DenseGNN)

- **架构**: DenseGNN with ASU (Asymmetric Unit) 支持
- **深度**: 5 层
- **隐藏单元**: 128
- **输入特征**:
  - 原子编号
  - 原子坐标偏移量
  - Voronoi ridge area
  - AGNI指纹 (128维)
  - 晶格电荷

- **输出**: 带隙预测 (mbj方法计算)

### 训练配置

- **损失函数**: Mean Absolute Error (MAE)
- **优化器**: Adam with Exponential Decay
  - 初始学习率: 0.001
  - 衰减步数: 5800
  - 衰减率: 0.5

- **学习率调度器**: Linear 调度
  - 起始: 0.001
  - 结束: 1e-05
  - Epoch范围: 100-300

- **交叉验证**: 5折 KFold
- **标准化**: StandardScaler (mean=0, std=1)

### 图表示

- **预处理器**: VoronoiUnitCell
- **最小 ridge area**: 0.01

## 📁 项目文件结构

```
DenseGNN/
├── kgcnn/                          # 本地 kgcnn 包
│   ├── __init__.py                 # ✓ 已创建
│   ├── io/                         # ✓ 已复制
│   ├── literature/DenseGNN/        # DenseGNN 模型
│   └── data/datasets/              # JARVIS 数据集类
├── training/
│   ├── train_crystal.py            # ✓ 已修复
│   └── hyper/
│       └── hyper_jarvis_mbj_bandgap.py  # ✓ 已创建
├── run_training_simple.sh          # ✓ 已创建
├── start_training.py               # ✓ 已创建
├── test_setup.py                   # ✓ 已创建
├── download_dataset.py             # 数据集下载辅助脚本
└── TRAINING_SUMMARY.md             # 本文档
```

## 🔧 故障排除

### 问题: 模块导入错误

确保设置了 PYTHONPATH:
```bash
export PYTHONPATH=/home/user/DenseGNN:$PYTHONPATH
```

### 问题: TensorFlow 警告

TensorFlow 的 CUDA 警告可以忽略（CPU 模式）。可以通过以下方式减少日志输出：
```bash
export TF_CPP_MIN_LOG_LEVEL=2
```

### 问题: TensorFlow Addons 不兼容

已在训练脚本中注释掉相关导入。如果遇到问题，确认 `train_crystal.py:10` 已被注释。

## 📈 预期结果

训练完成后，结果将保存在：
```
training/results/
```

包括：
- 训练历史 (loss, metrics)
- 模型检查点
- 预测结果
- 可视化图表

## 🎯 下一步

1. **准备数据集** - 下载并放置 JARVIS mbj_bandgap 数据
2. **验证设置** - 运行 `python test_setup.py`（需先准备数据）
3. **开始训练** - 运行 `./run_training_simple.sh`
4. **监控进度** - 查看训练日志和 `training/results/` 目录
5. **评估模型** - 查看交叉验证结果和预测性能

## 📚 参考资料

- **论文**: Du, H., et al. "DenseGNN: universal and scalable deeper graph neural networks for high-performance property prediction in crystals and molecules." npj Computational Materials 10, 292 (2024).
- **JARVIS**: https://jarvis.nist.gov/
- **kgcnn**: https://github.com/aimat-lab/gcnn_keras

---

**状态**: 环境已配置完成 ✅ | 等待数据集准备 ⏳
