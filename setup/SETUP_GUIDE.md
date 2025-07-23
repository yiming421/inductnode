# 🚀 InductNode Auto-Setup Guide

## 📋 Quick Start

### 1. **准备配置文件**
```bash
# 复制配置模板
cp setup_config_template.sh setup_config.sh

# 编辑配置文件，填入你的信息
nano setup_config.sh
```

### 2. **必填配置项**
在 `setup_config.sh` 中填入：
```bash
export GITHUB_REPO_URL="https://github.com/YOUR_USERNAME/inductnode.git"
export GITHUB_USER="YOUR_USERNAME"
```

### 3. **运行安装脚本**
```bash
# 加载配置
source setup_config.sh

# 运行安装
chmod +x setup_inductnode.sh
./setup_inductnode.sh
```

## 🎯 使用场景

### 🔄 **场景1: 首次安装**
```bash
# 1. 配置
export GITHUB_REPO_URL="https://github.com/username/inductnode.git"
export GITHUB_USER="username"

# 2. 运行脚本
./setup_inductnode.sh
```

### 🔄 **场景2: 更新现有项目**
```bash
# 脚本会自动检测现有安装并更新
./setup_inductnode.sh
```

### 🔄 **场景3: 自定义安装路径**
```bash
export INSTALL_DIR="/opt/ml-projects"
export CONDA_ENV_NAME="my-inductnode"
./setup_inductnode.sh
```

## 📁 脚本功能特性

### ✅ **自动处理的任务**
- [x] 检查系统依赖 (conda/mamba, git, CUDA)
- [x] 🔐 自动配置Git用户信息和认证
- [x] 🔑 引导设置GitHub认证 (Token/SSH)
- [x] 克隆GitHub仓库并配置远程连接
- [x] 基于 `env.yaml` 创建conda环境
- [x] 安装项目依赖 (setup.py/pyproject.toml)
- [x] 验证安装状态
- [x] 创建快速激活脚本

### 🛡️ **智能保护机制**
- [x] 自动备份本地更改 (git stash)
- [x] 检测现有环境并询问是否更新
- [x] 验证关键组件安装状态
- [x] 彩色输出和详细日志

### 🎨 **便利功能**
- [x] 交互式配置引导
- [x] 一键激活脚本 (`activate_env.sh`)
- [x] 详细的使用说明
- [x] 错误处理和回滚

## 🔧 高级配置

### 📝 **配置文件选项**
```bash
# 基础配置
GITHUB_REPO_URL=""     # GitHub仓库地址
GITHUB_USER=""         # GitHub用户名
INSTALL_DIR=""         # 安装目录
CONDA_ENV_NAME=""      # conda环境名称

# 高级选项
FORCE_REINSTALL=true   # 强制重新安装
SKIP_VERIFICATION=true # 跳过验证步骤
AUTO_ACTIVATE=true     # 安装后自动激活
```

### 🚀 **使用技巧**

#### 快速重新安装
```bash
export FORCE_REINSTALL=true
./setup_inductnode.sh
```

#### 批量安装 (多个环境)
```bash
# 环境1
export CONDA_ENV_NAME="inductnode-dev"
./setup_inductnode.sh

# 环境2  
export CONDA_ENV_NAME="inductnode-prod"
./setup_inductnode.sh
```

#### 离线模式配置
```bash
# 如果已有本地仓库
export SKIP_CLONE=true
export PROJECT_PATH="/path/to/existing/inductnode"
./setup_inductnode.sh
```

## 🐛 故障排除

### ❌ **常见问题**

1. **Conda未找到**
   ```bash
   # 安装miniconda
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

2. **Git认证失败**
   ```bash
   # 脚本会自动处理Git配置，但如果仍有问题：
   
   # 方法1: 重新运行脚本，选择认证方式
   ./setup_inductnode.sh
   
   # 方法2: 手动检查Git配置
   git config --global --list
   
   # 方法3: 使用Personal Access Token
   # 脚本会引导您创建和配置GitHub token
   ```

3. **环境冲突**
   ```bash
   # 删除现有环境
   conda env remove -n inductnode
   # 重新运行脚本
   ./setup_inductnode.sh
   ```

### 🔍 **调试模式**
```bash
# 启用详细输出
set -x
./setup_inductnode.sh
```

## 📈 完成后的工作流

### 🔄 **日常开发流程**
```bash
# 1. 激活环境
source activate_env.sh
# 或者
conda activate inductnode

# 2. 拉取最新代码
git pull

# 3. 运行实验
wandb sweep sweep.yaml

# 4. 提交更改
git add .
git commit -m "your message"
git push
```

### 🎯 **项目更新流程**
```bash
# 更新代码和环境
./setup_inductnode.sh

# 或手动更新
conda activate inductnode
git pull
conda env update -f env.yaml
```

---
**🎉 现在你就有了一个完全自动化的InductNode项目设置流程！**
