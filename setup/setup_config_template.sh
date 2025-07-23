# =============================================================================
# InductNode Setup Configuration
# =============================================================================
# Copy this file to 'setup_config.sh' and fill in your specific values
# Then source it before running the setup script: source setup_config.sh

# 🔗 GitHub Repository Configuration
export GITHUB_REPO_URL="https://github.com/yiming421/inductnode.git"  # e.g., "https://github.com/username/inductnode.git"
export GITHUB_USER="yiming421"      # e.g., "your-github-username"

# 📁 Installation Paths
export INSTALL_DIR="$HOME"           # Where to install the project
export PROJECT_NAME="inductnode"     # Project folder name
export CONDA_ENV_NAME="inductnode"   # Conda environment name

# 🐍 Environment Configuration
export ENV_FILE="env.yml"           # Name of the conda environment file

# 🎯 Optional: Pre-configure git
export GIT_USER_EMAIL="2200013081@stu.pku.edu.cn"            # Your git email (will auto-configure if provided)
export GITHUB_TOKEN=""               # Your GitHub personal access token (optional)

# 🔧 Advanced Options
export FORCE_REINSTALL=false        # Set to true to force clean reinstall
export SKIP_VERIFICATION=false      # Set to true to skip installation verification
export AUTO_ACTIVATE=true           # Set to true to auto-activate environment after setup

# 💡 Usage Examples:
# 
# Basic usage:
# GITHUB_REPO_URL="https://github.com/username/inductnode.git"
# GITHUB_USER="username"
# GIT_USER_EMAIL="username@example.com"
# 
# With GitHub token (recommended for private repos):
# GITHUB_REPO_URL="https://github.com/username/inductnode.git" 
# GITHUB_USER="username"
# GIT_USER_EMAIL="username@example.com"
# GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# 
# Custom installation:
# INSTALL_DIR="/opt/ml-projects"
# CONDA_ENV_NAME="my-inductnode-env"
# 
# Development mode:
# FORCE_REINSTALL=true
# SKIP_VERIFICATION=true
