# DenseGNN 安装指南

## 推荐安装方法（最简单）

### 方法一：分步安装（推荐）

```bash
# 1. 创建虚拟环境
conda create -n densegnn python=3.9
conda activate densegnn

# 2. 先安装需要编译的包（使用 conda，避免编译问题）
conda install -c conda-forge numpy scipy pandas matplotlib scikit-learn networkx

# 3. 安装 pymatgen（conda 提供预编译版本）
conda install -c conda-forge pymatgen

# 4. 安装 rdkit
conda install -c conda-forge rdkit

# 5. 安装 tensorflow
pip install tensorflow==2.9.0

# 6. 安装其他依赖
pip install tensorflow-addons==0.18.0 keras-tuner==1.1.3 requests==2.28.1 \
    sympy==1.11.1 pyyaml==6.0 ase==3.22.1 click==7.1.2 brotli==1.0.9 \
    pyxtal==0.5.5 h5py==3.9.0

# 7. 安装 kgcnn 包
cd /path/to/DenseGNN
pip install -e .
```

### 方法二：使用更新的依赖版本

如果您不严格需要最小版本，可以使用更新的版本（更容易安装）：

```bash
# 创建环境
conda create -n densegnn python=3.9
conda activate densegnn

# 使用 conda 安装主要依赖
conda install -c conda-forge numpy scipy pandas matplotlib scikit-learn \
    networkx pymatgen rdkit tensorflow

# 使用 pip 安装其他依赖
pip install tensorflow-addons keras-tuner requests sympy pyyaml ase click \
    brotli pyxtal h5py

# 安装 kgcnn
cd /path/to/DenseGNN
pip install -e .
```

### 方法三：仅使用 pip（需要编译工具）

如果您坚持使用 pip 和最小版本，需要先安装编译工具：

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

**CentOS/RHEL:**
```bash
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```

**macOS:**
```bash
xcode-select --install
```

**然后安装依赖:**
```bash
pip install -r requirements_min.txt
pip install -e .
```

## 常见问题解决

### 1. pymatgen 编译失败

**解决方案A:** 使用 conda 安装
```bash
conda install -c conda-forge pymatgen
```

**解决方案B:** 使用更新的版本（有预编译包）
```bash
pip install pymatgen>=2023.0.0
```

### 2. rdkit 安装失败

**解决方案:** 使用 conda 安装
```bash
conda install -c conda-forge rdkit
```

### 3. tensorflow GPU 支持

如果需要 GPU 支持：

1. 检查 CUDA 版本：`nvidia-smi`
2. 根据 CUDA 版本选择合适的 tensorflow：
   - CUDA 11.2: `pip install tensorflow==2.9.0`
   - 查看兼容性: https://www.tensorflow.org/install/source#gpu

3. 安装 cuDNN（需要与 CUDA 版本匹配）

## 验证安装

```bash
# 测试导入
python -c "import kgcnn; print('kgcnn version:', kgcnn.__version__)"
python -c "import tensorflow; print('TensorFlow version:', tensorflow.__version__)"
python -c "import pymatgen; print('pymatgen version:', pymatgen.__version__)"
python -c "import rdkit; print('RDKit imported successfully')"
```

## 最小化依赖安装（不推荐）

如果您只想测试核心功能，可以只安装以下必需包：

```bash
pip install numpy tensorflow keras rdkit pymatgen
pip install -e . --no-deps
```

注意：这可能导致某些功能不可用。

## 推荐的完整环境配置

```bash
# 创建新环境
conda create -n densegnn python=3.9 -y
conda activate densegnn

# 使用 conda 安装科学计算包（避免编译问题）
conda install -c conda-forge -y \
    numpy=1.23 scipy=1.9 pandas=1.5 matplotlib=3.6 \
    scikit-learn=1.1 networkx=2.8 sympy=1.11 \
    pymatgen rdkit openbabel ase

# 使用 pip 安装深度学习和其他包
pip install tensorflow==2.9.0 tensorflow-addons==0.18.0 \
    keras-tuner==1.1.3 pyyaml==6.0 pyxtal==0.5.5 \
    h5py==3.9.0 click==7.1.2 brotli==1.0.9 requests==2.28.1

# 安装 kgcnn
cd /path/to/DenseGNN
pip install -e .

echo "安装完成！"
```

## 故障排除

如果遇到其他问题：

1. 检查 Python 版本：`python --version` （应该是 3.9 或更高）
2. 更新 pip：`pip install --upgrade pip setuptools wheel`
3. 清除 pip 缓存：`pip cache purge`
4. 如果特定包失败，尝试使用 conda 安装该包
5. 查看详细错误信息：`pip install -v <package>`

## 依赖版本说明

`requirements_min.txt` 文件中 pymatgen 使用 `>=2022.11.7` 而不是 `==2022.11.7`，
是因为精确版本需要从源码编译，而更新的版本提供了预编译的二进制包，更容易安装且完全兼容。
