# 切换到 PyTorch Backend

kgcnn 使用 Keras 3 的多后端系统，可以无缝切换到 PyTorch！

## 1. 安装 PyTorch (CUDA 11.8)

```bash
# 卸载旧的 TensorFlow (可选)
pip uninstall tensorflow tensorflow-addons -y

# 安装 PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证 PyTorch GPU 支持
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

## 2. 设置 Keras Backend 为 PyTorch

有三种方法设置后端：

### 方法 A: 环境变量（推荐）

```bash
export KERAS_BACKEND=torch

# 验证后端
python -c "import keras; print('Keras backend:', keras.backend.backend())"
```

### 方法 B: 配置文件

创建或编辑 `~/.keras/keras.json`:

```json
{
    "backend": "torch",
    "image_data_format": "channels_last",
    "floatx": "float32",
    "epsilon": 1e-07
}
```

### 方法 C: 代码中设置（在任何 import 之前）

```python
import os
os.environ['KERAS_BACKEND'] = 'torch'

import keras
import kgcnn
```

## 3. 运行训练

```bash
# 设置 PyTorch backend
export KERAS_BACKEND=torch

# 启用 CUDA
export CUDA_VISIBLE_DEVICES=0

# 运行训练
cd /home/user/DenseGNN
python training/train_crystal.py --hyper training/hyper/hyper_mp_log_kvrh.py
```

## 4. PyTorch 的优势

- ✅ **更好的 GPU 支持**: RTX 4090 完全支持
- ✅ **更快的编译**: 动态图模式，无需 XLA
- ✅ **更简单的调试**: 即时执行模式
- ✅ **更新的 CUDA 支持**: 支持最新的 NVIDIA GPU

## 5. 验证完整设置

```bash
# 完整验证脚本
python -c "
import os
os.environ['KERAS_BACKEND'] = 'torch'

import keras
import torch
import kgcnn

print('=' * 50)
print('Backend Configuration')
print('=' * 50)
print(f'Keras backend: {keras.backend.backend()}')
print(f'kgcnn version: {kgcnn.__version__ if hasattr(kgcnn, \"__version__\") else \"unknown\"}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
print('=' * 50)
"
```

## 6. 故障排除

### GPU 未检测到

```bash
# 检查 CUDA 库路径
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# 检查 NVIDIA 驱动
nvidia-smi
```

### Keras backend 未切换

确保在 import keras 之前设置环境变量：

```python
import os
os.environ['KERAS_BACKEND'] = 'torch'  # 必须在 import keras 之前！

import keras  # 现在才 import
```

### 依赖冲突

如果遇到依赖问题：

```bash
# 重新安装 Keras 3
pip install --upgrade keras>=3.0.0

# 确保没有旧的 tf.keras
pip uninstall keras-tuner -y  # keras-tuner 依赖 tensorflow
```

## 7. 性能优化

```bash
# 启用 PyTorch 优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_USE_CUDA_DSA=1

# 启用 cuDNN benchmark
python -c "import torch; torch.backends.cudnn.benchmark = True"
```

## 8. 完整训练命令

```bash
# 一键运行（推荐）
cd /home/user/DenseGNN
KERAS_BACKEND=torch CUDA_VISIBLE_DEVICES=0 python training/train_crystal.py --hyper training/hyper/hyper_mp_log_kvrh.py
```
