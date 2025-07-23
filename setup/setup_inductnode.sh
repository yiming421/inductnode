#!/bin/bash

# =============================================================================
# InductNode Project Auto-Setup Script
# =============================================================================
# This script automatically sets up the InductNode project with proper
# conda environment, GitHub integration, and all dependencies.
# =============================================================================

set -e  # Exit on any error

# 🎨 Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 📁 Configuration Variables (FILL THESE IN)
# =============================================================================
GITHUB_REPO_URL=""  # TODO: Fill in your GitHub repo URL
GITHUB_USER=""      # TODO: Fill in your GitHub username  
PROJECT_NAME="inductnode"
CONDA_ENV_NAME="inductnode"
INSTALL_DIR="$HOME"
PROJECT_PATH="$INSTALL_DIR/$PROJECT_NAME"
ENV_FILE="env.yaml"

# 🔧 Function Definitions
# =============================================================================

print_banner() {
    echo -e "${PURPLE}"
    echo "════════════════════════════════════════════════════════════════"
    echo "🚀 InductNode Project Auto-Setup Script"
    echo "════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

check_dependencies() {
    log_step "Checking system dependencies..."
    
    # Check if conda/mamba is installed
    if command -v conda &> /dev/null; then
        log_success "Conda found: $(conda --version)"
        CONDA_CMD="conda"
    elif command -v mamba &> /dev/null; then
        log_success "Mamba found: $(mamba --version)"
        CONDA_CMD="mamba"
    else
        log_error "Neither conda nor mamba found. Please install Anaconda/Miniconda first."
        echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    # Check if git is installed
    if command -v git &> /dev/null; then
        log_success "Git found: $(git --version)"
    else
        log_error "Git not found. Please install git first."
        exit 1
    fi
    
    # Check if CUDA is available (optional)
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA drivers found: $(nvidia-smi --version | head -1)"
    else
        log_warning "NVIDIA drivers not found. CPU-only mode will be used."
    fi
}

prompt_for_config() {
    log_step "Gathering configuration information..."
    
    if [ -z "$GITHUB_REPO_URL" ]; then
        echo -e "${YELLOW}Please enter your GitHub repository URL:${NC}"
        echo "Example: https://github.com/username/inductnode.git"
        read -p "> " GITHUB_REPO_URL
    fi
    
    if [ -z "$GITHUB_USER" ]; then
        echo -e "${YELLOW}Please enter your GitHub username:${NC}"
        read -p "> " GITHUB_USER
    fi
    
    echo -e "${YELLOW}Enter installation directory (default: $INSTALL_DIR):${NC}"
    read -p "> " user_install_dir
    if [ ! -z "$user_install_dir" ]; then
        INSTALL_DIR="$user_install_dir"
        PROJECT_PATH="$INSTALL_DIR/$PROJECT_NAME"
    fi
    
    echo -e "${YELLOW}Enter conda environment name (default: $CONDA_ENV_NAME):${NC}"
    read -p "> " user_env_name
    if [ ! -z "$user_env_name" ]; then
        CONDA_ENV_NAME="$user_env_name"
    fi
    
    log_info "Configuration summary:"
    echo "  📁 Install Directory: $INSTALL_DIR"
    echo "  🐍 Conda Environment: $CONDA_ENV_NAME"
    echo "  🔗 GitHub Repository: $GITHUB_REPO_URL"
    echo "  👤 GitHub User: $GITHUB_USER"
}

setup_git_config() {
    log_step "Configuring Git settings..."
    
    # Check if global git config exists
    local current_git_name=$(git config --global user.name 2>/dev/null)
    local current_git_email=$(git config --global user.email 2>/dev/null)
    
    if [ -z "$current_git_name" ] || [ -z "$current_git_email" ]; then
        log_info "Git user configuration not found. Setting up Git..."
        
        # Use GitHub username if no git name is set
        if [ -z "$current_git_name" ]; then
            if [ ! -z "$GITHUB_USER" ]; then
                log_info "Setting Git username to: $GITHUB_USER"
                git config --global user.name "$GITHUB_USER"
            else
                echo -e "${YELLOW}Please enter your Git username:${NC}"
                read -p "> " git_username
                git config --global user.name "$git_username"
            fi
        fi
        
        # Set up email
        if [ -z "$current_git_email" ]; then
            if [ ! -z "$GIT_USER_EMAIL" ]; then
                log_info "Setting Git email to: $GIT_USER_EMAIL"
                git config --global user.email "$GIT_USER_EMAIL"
            else
                echo -e "${YELLOW}Please enter your Git email:${NC}"
                read -p "> " git_email
                git config --global user.email "$git_email"
            fi
        fi
        
        log_success "Git configuration completed"
    else
        log_success "Git already configured: $current_git_name <$current_git_email>"
    fi
    
    # Set up Git credential helper for HTTPS
    setup_git_credentials
}

setup_git_credentials() {
    log_info "Setting up Git credential helper..."
    
    # Check if credential helper is already configured
    local current_helper=$(git config --global credential.helper 2>/dev/null)
    
    if [ -z "$current_helper" ]; then
        # Set up credential helper based on system
        if command -v git-credential-manager &> /dev/null; then
            log_info "Setting up Git Credential Manager..."
            git config --global credential.helper manager
        elif [ -f "/usr/share/doc/git/contrib/credential/libsecret/git-credential-libsecret" ]; then
            log_info "Setting up libsecret credential helper..."
            git config --global credential.helper /usr/share/doc/git/contrib/credential/libsecret/git-credential-libsecret
        else
            log_info "Setting up cache credential helper..."
            git config --global credential.helper 'cache --timeout=3600'
        fi
        
        log_success "Git credential helper configured"
    else
        log_success "Git credential helper already configured: $current_helper"
    fi
    
    # Offer to set up SSH keys if HTTPS authentication might be problematic
    if [[ "$GITHUB_REPO_URL" == *"https://github.com"* ]]; then
        setup_github_auth_method
    fi
}

setup_github_auth_method() {
    log_info "GitHub authentication setup..."
    
    echo -e "${CYAN}Choose your preferred GitHub authentication method:${NC}"
    echo "1. Personal Access Token (recommended for HTTPS)"
    echo "2. SSH Key (recommended for SSH)"
    echo "3. Skip (use existing authentication)"
    read -p "Enter choice (1-3): " auth_choice
    
    case $auth_choice in
        1)
            setup_github_token
            ;;
        2)
            setup_github_ssh
            ;;
        3)
            log_info "Skipping GitHub authentication setup"
            ;;
        *)
            log_warning "Invalid choice. Skipping authentication setup."
            ;;
    esac
}

setup_github_token() {
    echo -e "${YELLOW}Setting up GitHub Personal Access Token...${NC}"
    echo "📋 Steps to create a token:"
    echo "1. Go to https://github.com/settings/tokens"
    echo "2. Click 'Generate new token (classic)'"
    echo "3. Select scopes: repo, workflow, write:packages"
    echo "4. Copy the generated token"
    echo ""
    echo -e "${YELLOW}Paste your GitHub token (it will be hidden):${NC}"
    read -s github_token
    
    if [ ! -z "$github_token" ]; then
        # Store token securely using git credential helper
        echo "protocol=https
host=github.com
username=$GITHUB_USER
password=$github_token" | git credential approve
        
        log_success "GitHub token configured successfully"
    else
        log_warning "No token provided. You may need to authenticate manually."
    fi
}

setup_github_ssh() {
    log_info "Setting up SSH key for GitHub..."
    
    local ssh_key_path="$HOME/.ssh/id_rsa"
    local ssh_pub_key_path="$ssh_key_path.pub"
    
    # Check if SSH key already exists
    if [ -f "$ssh_key_path" ]; then
        log_success "SSH key already exists at $ssh_key_path"
    else
        echo -e "${YELLOW}Enter your email for SSH key generation:${NC}"
        read -p "> " ssh_email
        
        log_info "Generating SSH key..."
        ssh-keygen -t rsa -b 4096 -C "$ssh_email" -f "$ssh_key_path" -N ""
        
        # Start ssh-agent and add key
        eval "$(ssh-agent -s)"
        ssh-add "$ssh_key_path"
        
        log_success "SSH key generated at $ssh_key_path"
    fi
    
    # Display public key for copying to GitHub
    if [ -f "$ssh_pub_key_path" ]; then
        echo -e "${CYAN}📋 Copy this SSH public key to GitHub:${NC}"
        echo "Go to: https://github.com/settings/keys"
        echo ""
        cat "$ssh_pub_key_path"
        echo ""
        echo -e "${YELLOW}Press Enter after adding the key to GitHub...${NC}"
        read -p ""
        
        # Test SSH connection
        log_info "Testing SSH connection to GitHub..."
        if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
            log_success "SSH authentication successful"
            
            # Convert HTTPS URL to SSH if needed
            if [[ "$GITHUB_REPO_URL" == *"https://github.com"* ]]; then
                local ssh_url=$(echo "$GITHUB_REPO_URL" | sed 's|https://github.com/|git@github.com:|')
                echo -e "${YELLOW}Convert to SSH URL? Current: $GITHUB_REPO_URL${NC}"
                echo -e "${YELLOW}Would become: $ssh_url${NC}"
                read -p "Convert to SSH? (y/n): " convert_ssh
                if [[ $convert_ssh =~ ^[Yy]$ ]]; then
                    GITHUB_REPO_URL="$ssh_url"
                    log_success "Repository URL updated to SSH"
                fi
            fi
        else
            log_warning "SSH authentication test failed. Please check your key setup."
        fi
    fi
}

clone_or_update_repo() {
    log_step "Setting up repository..."
    
    if [ -d "$PROJECT_PATH" ]; then
        log_info "Project directory exists. Checking if it's a git repository..."
        
        if [ -d "$PROJECT_PATH/.git" ]; then
            log_info "Updating existing repository..."
            cd "$PROJECT_PATH"
            
            # Stash any local changes
            if ! git diff --quiet || ! git diff --staged --quiet; then
                log_warning "Local changes detected. Stashing them..."
                git stash push -m "Auto-stash before update $(date)"
            fi
            
            # Pull latest changes
            git pull origin main || git pull origin master
            log_success "Repository updated successfully"
        else
            log_warning "Directory exists but is not a git repository. Removing..."
            rm -rf "$PROJECT_PATH"
            clone_fresh_repo
        fi
    else
        clone_fresh_repo
    fi
}

clone_fresh_repo() {
    log_info "Cloning repository from $GITHUB_REPO_URL..."
    cd "$INSTALL_DIR"
    
    # Clone the repository
    git clone "$GITHUB_REPO_URL" "$PROJECT_NAME"
    cd "$PROJECT_PATH"
    
    # Set up remote tracking
    git remote set-url origin "$GITHUB_REPO_URL"
    
    # Configure git if not already done
    setup_git_config
    
    log_success "Repository cloned and configured successfully"
}

setup_conda_env() {
    log_step "Setting up conda environment..."
    
    # Check if environment file exists
    if [ ! -f "$PROJECT_PATH/$ENV_FILE" ]; then
        log_error "Environment file $ENV_FILE not found in the repository!"
        log_info "Please ensure your repository contains an $ENV_FILE file."
        exit 1
    fi
    
    # Check if environment already exists
    if $CONDA_CMD env list | grep -q "^$CONDA_ENV_NAME "; then
        log_warning "Conda environment '$CONDA_ENV_NAME' already exists."
        echo -e "${YELLOW}Do you want to update it? (y/n):${NC}"
        read -p "> " update_env
        
        if [[ $update_env =~ ^[Yy]$ ]]; then
            log_info "Updating conda environment..."
            $CONDA_CMD env update -n "$CONDA_ENV_NAME" -f "$PROJECT_PATH/$ENV_FILE"
        fi
    else
        log_info "Creating new conda environment from $ENV_FILE..."
        $CONDA_CMD env create -n "$CONDA_ENV_NAME" -f "$PROJECT_PATH/$ENV_FILE"
    fi
    
    log_success "Conda environment '$CONDA_ENV_NAME' is ready"
}

install_project() {
    log_step "Installing project in development mode..."
    
    # Activate conda environment and install project
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    
    cd "$PROJECT_PATH"
    
    # Install project in development mode if setup.py exists
    if [ -f "setup.py" ]; then
        log_info "Installing project with setup.py..."
        pip install -e .
    elif [ -f "pyproject.toml" ]; then
        log_info "Installing project with pyproject.toml..."
        pip install -e .
    else
        log_warning "No setup.py or pyproject.toml found. Skipping project installation."
    fi
    
    # Install additional requirements if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        log_info "Installing additional requirements..."
        pip install -r requirements.txt
    fi
    
    log_success "Project installation completed"
}

verify_installation() {
    log_step "Verifying installation..."
    
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    
    # Test basic imports
    cd "$PROJECT_PATH"
    
    log_info "Testing Python imports..."
    python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA devices: {torch.cuda.device_count()}')
except ImportError:
    print('PyTorch not found')

try:
    import torch_geometric
    print(f'PyG version: {torch_geometric.__version__}')
except ImportError:
    print('PyTorch Geometric not found')

try:
    import wandb
    print(f'Wandb version: {wandb.__version__}')
except ImportError:
    print('Wandb not found')
" 2>/dev/null || log_warning "Some imports failed. Check your environment setup."
    
    log_success "Installation verification completed"
}

create_activation_script() {
    log_step "Creating activation script..."
    
    cat > "$PROJECT_PATH/activate_env.sh" << EOF
#!/bin/bash
# Quick activation script for InductNode environment

echo "🚀 Activating InductNode environment..."
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV_NAME
cd "$PROJECT_PATH"

echo "✅ Environment activated!"
echo "📁 Current directory: \$(pwd)"
echo "🐍 Python version: \$(python --version)"
echo "🔥 PyTorch version: \$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not found')"

# Start a new shell with the environment activated
exec bash
EOF
    
    chmod +x "$PROJECT_PATH/activate_env.sh"
    log_success "Activation script created at $PROJECT_PATH/activate_env.sh"
}

print_final_instructions() {
    echo -e "${GREEN}"
    echo "════════════════════════════════════════════════════════════════"
    echo "🎉 Installation completed successfully!"
    echo "════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    
    echo -e "${CYAN}Next steps:${NC}"
    echo "1. Activate your environment:"
    echo -e "   ${YELLOW}conda activate $CONDA_ENV_NAME${NC}"
    echo ""
    echo "2. Navigate to project directory:"
    echo -e "   ${YELLOW}cd $PROJECT_PATH${NC}"
    echo ""
    echo "3. Or use the quick activation script:"
    echo -e "   ${YELLOW}$PROJECT_PATH/activate_env.sh${NC}"
    echo ""
    echo -e "${CYAN}Useful commands:${NC}"
    echo "• Update repository: git pull"
    echo "• Push changes: git add . && git commit -m 'message' && git push"
    echo "• Update environment: $CONDA_CMD env update -n $CONDA_ENV_NAME -f $ENV_FILE"
    echo "• Run sweep: wandb sweep sweep.yaml"
    echo ""
    echo -e "${GREEN}Happy coding! 🚀${NC}"
}

# 🚀 Main Execution Flow
# =============================================================================

main() {
    print_banner
    
    # Check system dependencies
    check_dependencies
    
    # Get user configuration
    prompt_for_config
    
    # Confirm before proceeding
    echo -e "${YELLOW}Proceed with installation? (y/n):${NC}"
    read -p "> " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        log_info "Installation cancelled by user."
        exit 0
    fi
    
    # Execute setup steps
    clone_or_update_repo
    setup_conda_env
    install_project
    verify_installation
    create_activation_script
    
    # Show final instructions
    print_final_instructions
}

# Run main function
main "$@"
