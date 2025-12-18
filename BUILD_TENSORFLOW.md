# 从源码编译 TensorFlow 支持 RTX 5090 (Compute Capability 12.0)

## 前置要求

- CUDA 12.x (您已有)
- Python 3.10
- Bazel (TensorFlow 构建工具)
- 大量磁盘空间 (~50GB)
- 编译时间：4-8 小时

## 步骤 1: 安装 Bazelisk (Bazel 版本管理器)

```bash
# 下载 Bazelisk
wget https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel

# 验证
bazel --version
```

## 步骤 2: 克隆 TensorFlow 源码

```bash
cd ~
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow

# 使用最新稳定版本或开发分支
git checkout master  # 或 r2.20
```

## 步骤 3: 配置编译选项

```bash
cd ~/tensorflow

# 配置构建
./configure

# 重要配置项（交互式）：
# - Python location: /public/home/ghzhang/.conda/envs/cogn/bin/python
# - Python library path: (自动检测)
# - Do you wish to build TensorFlow with CUDA support? [y/N]: y
# - CUDA SDK version: 12.x (您的版本)
# - cuDNN version: 8.x (您的版本)
# - Compute capabilities: 12.0  # <-- 关键！RTX 5090
# - Use clang as CUDA compiler? [Y/n]: n
# - GCC host compiler: /usr/bin/gcc
# - Optimization flags: -march=native -Wno-sign-compare
```

### 自动化配置脚本

创建 `.tf_configure.bazelrc` 或使用环境变量：

```bash
export TF_NEED_CUDA=1
export TF_CUDA_VERSION=12
export TF_CUDNN_VERSION=8
export TF_CUDA_COMPUTE_CAPABILITIES=12.0  # RTX 5090
export CUDA_TOOLKIT_PATH=/usr/local/cuda-12
export CUDNN_INSTALL_PATH=/usr/local/cuda-12
export TF_CUDA_CLANG=0
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
export PYTHON_BIN_PATH=/public/home/ghzhang/.conda/envs/cogn/bin/python

# 运行配置
./configure
```

## 步骤 4: 编译 TensorFlow

```bash
cd ~/tensorflow

# 使用 Bazel 编译 pip 包
# --config=cuda: 启用 CUDA 支持
# --config=opt: 优化构建
# -c opt: 优化标志
bazel build --config=cuda --config=opt -c opt //tensorflow/tools/pip_package:build_pip_package

# 编译时间：4-8 小时，取决于 CPU 核心数
# 使用多核加速：
bazel build --config=cuda --config=opt -c opt --jobs=32 //tensorflow/tools/pip_package:build_pip_package
```

### 可选：减少编译时间的选项

```bash
# 只编译核心功能（更快）
bazel build --config=cuda --config=opt \
  --define=no_tensorflow_py_deps=true \
  --jobs=32 \
  //tensorflow/tools/pip_package:build_pip_package
```

## 步骤 5: 生成 Wheel 包

```bash
# 创建 wheel 包
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# 查看生成的包
ls -lh /tmp/tensorflow_pkg/tensorflow-*.whl
```

## 步骤 6: 安装编译好的 TensorFlow

```bash
# 卸载旧版本
pip uninstall tensorflow tf-nightly -y

# 安装新编译的版本
pip install /tmp/tensorflow_pkg/tensorflow-*.whl

# 验证
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
print('GPU devices:', tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'):
    print('GPU name:', tf.config.list_physical_devices('GPU')[0])
"
```

## 步骤 7: 测试 GPU

```bash
# 测试 GPU 计算
python -c "
import tensorflow as tf
with tf.device('/GPU:0'):
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])
    c = tf.matmul(a, b)
    print('GPU computation successful!')
    print('Result shape:', c.shape)
"
```

## 步骤 8: 运行 DenseGNN 训练

```bash
cd /public/home/ghzhang/soft/DenseGNN-main

KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=0 python training/train_crystal.py \
  --hyper training/hyper/hyper_mp_log_kvrh.py \
  --category DenseGNN
```

---

## 故障排除

### 编译错误：内存不足

```bash
# 限制并行任务
bazel build --config=cuda --config=opt --jobs=8 --local_ram_resources=8192 //tensorflow/tools/pip_package:build_pip_package
```

### CUDA 库找不到

```bash
# 设置路径
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12/bin:$PATH
```

### Bazel 版本问题

```bash
# 使用特定 Bazel 版本
export USE_BAZEL_VERSION=6.5.0  # 根据 TensorFlow 版本调整
bazel --version
```

### 检查 compute capability 是否正确编译

```bash
# 检查编译的 compute capabilities
strings /tmp/tensorflow_pkg/tensorflow-*.whl | grep "compute_"
```

---

## 预计资源需求

- **磁盘空间**: ~50GB (源码 + 构建缓存)
- **内存**: 至少 16GB RAM (推荐 32GB)
- **CPU**: 越多越快 (32 核心 ~2-3 小时，8 核心 ~6-8 小时)
- **时间**: 首次编译 4-8 小时

---

## 快速验证脚本

```bash
#!/bin/bash
# verify_tf_build.sh

echo "=== TensorFlow Build Verification ==="

echo -e "\n1. TensorFlow Version:"
python -c "import tensorflow as tf; print(tf.__version__)"

echo -e "\n2. CUDA Support:"
python -c "import tensorflow as tf; print('Built with CUDA:', tf.test.is_built_with_cuda())"

echo -e "\n3. GPU Devices:"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

echo -e "\n4. Compute Capability:"
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print('GPU Details:')
    for gpu in gpus:
        print(f'  {gpu}')
"

echo -e "\n5. GPU Test:"
python -c "
import tensorflow as tf
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print('GPU computation successful!')
"
```

---

## 参考资料

- [TensorFlow Build from Source](https://www.tensorflow.org/install/source)
- [CUDA Compute Capabilities](https://developer.nvidia.com/cuda-gpus)
- [Bazel Build System](https://bazel.build/)
