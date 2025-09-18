#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Logging functions
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

log_header() {
    echo -e "${PURPLE}${BOLD}[SETUP]${NC} $1"
}

log_step() {
    echo -e "${CYAN}➤${NC} $1"
}

# Check Python version
check_python() {
    log_step "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    REQUIRED_VERSION="3.8"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        log_error "Python $PYTHON_VERSION found. Python $REQUIRED_VERSION or higher is required."
        exit 1
    fi
    
    log_success "Python $PYTHON_VERSION detected"
}


setup_venv() {
    log_step "Setting up virtual environment..."
    
    if [ -d ".venv" ]; then
        log_warning "Virtual environment already exists. Removing old one..."
        rm -rf .venv
    fi
    
    python3 -m venv .venv
    source .venv/bin/activate
    
    log_info "Upgrading pip..."
    pip install --upgrade pip > /dev/null 2>&1
    
    log_info "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt

    log_success "Virtual environment created and dependencies installed"
}


create_directories() {
    log_step "Creating project directories..."
    
    directories=(
        "data/trainingandtestdata"
        "data/toks"
        "training/weights"
        "training/inference"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: ${WHITE}$dir${NC}"
        else
            log_info "Directory already exists: ${WHITE}$dir${NC}"
        fi
    done
    
    log_success "Project directories created "
}


main() {
    clear
    echo -e "${PURPLE}${BOLD}"
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║           Sentiment Analysis Project Setup           ║"
    echo "║                  Twitter Dataset                     ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo
    
    check_python
    setup_venv
    create_directories
    log_header "Setup successful"
}

# Run main function
main "$@"