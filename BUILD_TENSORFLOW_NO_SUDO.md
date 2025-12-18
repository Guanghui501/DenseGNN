# 从源码编译 TensorFlow（无 sudo 权限）

## 步骤 1: 安装 Bazelisk 到用户目录

```bash
# 创建本地 bin 目录
mkdir -p ~/bin

# 下载 Bazelisk（无需 sudo）
cd ~/bin
wget https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
ln -s bazelisk-linux-amd64 bazel

# 添加到 PATH
echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
bazel --version
```

## 步骤 2: 克隆 TensorFlow

```bash
cd ~
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r2.20  # 或 master
```

## 步骤 3: 配置编译（自动化脚本）

```bash
cd ~/tensorflow

# 设置环境变量（无需交互）
export TF_NEED_CUDA=1
export TF_CUDA_VERSION=12
export TF_CUDNN_VERSION=8
export TF_CUDA_COMPUTE_CAPABILITIES=12.0  # RTX 5090 关键配置
export CUDA_TOOLKIT_PATH=/usr/local/cuda-12
export CUDNN_INSTALL_PATH=/usr/local/cuda-12
export TF_CUDA_CLANG=0
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
export PYTHON_BIN_PATH=/public/home/ghzhang/.conda/envs/cogn/bin/python
export TF_NEED_ROCM=0
export TF_NEED_OPENCL_SYCL=0
export TF_DOWNLOAD_CLANG=0
export TF_SET_ANDROID_WORKSPACE=0

# 运行配置（会自动使用环境变量）
./configure
```

## 步骤 4: 编译（本地缓存）

```bash
cd ~/tensorflow

# 设置 Bazel 使用本地缓存目录（不需要 sudo）
export TEST_TMPDIR=$HOME/.cache/bazel
mkdir -p $TEST_TMPDIR

# 编译 TensorFlow
bazel build \
  --config=cuda \
  --config=opt \
  --jobs=32 \
  --local_ram_resources=16384 \
  //tensorflow/tools/pip_package:build_pip_package

# 如果内存不足，减少并行数：
# bazel build --config=cuda --config=opt --jobs=8 //tensorflow/tools/pip_package:build_pip_package
```

## 步骤 5: 生成和安装 Wheel 包

```bash
# 生成 wheel 包到用户目录
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tensorflow_pkg

# 查看生成的包
ls -lh ~/tensorflow_pkg/tensorflow-*.whl

# 安装（pip install 不需要 sudo）
pip uninstall tensorflow tf-nightly -y
pip install ~/tensorflow_pkg/tensorflow-*.whl
```

## 步骤 6: 验证

```bash
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
print('GPU devices:', tf.config.list_physical_devices('GPU'))

# GPU 测试
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print('GPU computation successful!')
"
```

## 完整一键脚本（复制粘贴运行）

```bash
#!/bin/bash
set -e

echo "=== 安装 Bazelisk 到用户目录 ==="
mkdir -p ~/bin
cd ~/bin
wget -q https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
ln -sf bazelisk-linux-amd64 bazel
export PATH=$HOME/bin:$PATH

echo "=== 克隆 TensorFlow ==="
cd ~
if [ ! -d "tensorflow" ]; then
    git clone --depth=1 https://github.com/tensorflow/tensorflow.git
fi
cd tensorflow

echo "=== 配置编译选项 ==="
export TF_NEED_CUDA=1
export TF_CUDA_VERSION=12
export TF_CUDNN_VERSION=8
export TF_CUDA_COMPUTE_CAPABILITIES=12.0
export CUDA_TOOLKIT_PATH=/usr/local/cuda-12
export CUDNN_INSTALL_PATH=/usr/local/cuda-12
export TF_CUDA_CLANG=0
export PYTHON_BIN_PATH=$(which python)
export TF_NEED_ROCM=0
export TF_NEED_OPENCL_SYCL=0
export TF_DOWNLOAD_CLANG=0
export TF_SET_ANDROID_WORKSPACE=0
export TEST_TMPDIR=$HOME/.cache/bazel

./configure

echo "=== 开始编译（需要 4-8 小时）==="
~/bin/bazel build \
  --config=cuda \
  --config=opt \
  --jobs=32 \
  //tensorflow/tools/pip_package:build_pip_package

echo "=== 生成 Wheel 包 ==="
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tensorflow_pkg

echo "=== 安装 TensorFlow ==="
pip uninstall -y tensorflow tf-nightly
pip install ~/tensorflow_pkg/tensorflow-*.whl

echo "=== 验证安装 ==="
python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"

echo "=== 完成！==="
```

保存为 `build_tf.sh` 并运行：
```bash
chmod +x build_tf.sh
./build_tf.sh
```

---

## 故障排除（无 sudo）

### 磁盘空间不足

```bash
# 清理 Bazel 缓存
~/bin/bazel clean --expunge

# 使用更小的缓存
export TEST_TMPDIR=/path/to/large/disk/.cache/bazel
```

### 内存不足

```bash
# 减少并行编译任务
bazel build --config=cuda --config=opt --jobs=4 --local_ram_resources=8192 \
  //tensorflow/tools/pip_package:build_pip_package
```

### CUDA 路径问题

```bash
# 检查 CUDA 安装位置
ls -la /usr/local/cuda*

# 如果在其他位置，更新路径
export CUDA_TOOLKIT_PATH=/your/cuda/path
export CUDNN_INSTALL_PATH=/your/cuda/path
```

---

## 预计时间和资源

- **首次编译**: 4-8 小时（取决于 CPU 核心数）
- **磁盘空间**: ~50GB
- **内存**: 建议 16GB+
- **后续编译**: ~1-2 小时（有缓存）

---

## 快速验证检查清单

```bash
# 1. Bazel 是否安装
~/bin/bazel --version

# 2. Python 路径
which python

# 3. CUDA 版本
cat /usr/local/cuda/version.txt || nvcc --version

# 4. 磁盘空间
df -h ~

# 5. 内存
free -h
```
