# ============================================================================
# ProjectStroke Training - Makefile
# ============================================================================
# Provides convenient commands for development, testing, and training
#
# Usage:
#   make install      - Install dependencies
#   make test         - Run smoke tests
#   make validate     - Run system health check
#   make clean        - Clean cache files
#   make train        - Run example training
#   make help         - Show this help message
# ============================================================================

.PHONY: help install install-minimal install-gpu test validate clean clean-all train optuna seeds format lint

# Python interpreter
PYTHON := python3
PIP := $(PYTHON) -m pip

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m  # No Color

# ============================================================================
# Help
# ============================================================================

help:
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║          ProjectStroke Training - Available Commands              ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Installation:$(NC)"
	@echo "  make install           Install all dependencies (CPU version)"
	@echo "  make install-minimal   Install minimal deps (validation only)"
	@echo "  make install-gpu       Install with GPU support (CUDA 11.8)"
	@echo ""
	@echo "$(GREEN)Testing & Validation:$(NC)"
	@echo "  make test              Run CLI smoke tests"
	@echo "  make validate          Run system health check"
	@echo ""
	@echo "$(GREEN)Training:$(NC)"
	@echo "  make train             Run example training (MLP, 1 epoch)"
	@echo "  make train-xgb         Run XGBoost training"
	@echo "  make optuna            Run Optuna optimization (5 trials)"
	@echo "  make seeds             Run multi-seed experiment (3 seeds)"
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@echo "  make clean             Clean Python cache files"
	@echo "  make clean-all         Deep clean (cache + outputs)"
	@echo "  make format            Format code with black (if installed)"
	@echo "  make lint              Lint code with flake8 (if installed)"
	@echo ""
	@echo "$(GREEN)Info:$(NC)"
	@echo "  make help              Show this help message"
	@echo ""

# ============================================================================
# Installation
# ============================================================================

install:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✅ Installation complete!$(NC)"
	@echo ""
	@echo "To verify, run: make validate"

install-minimal:
	@echo "$(BLUE)Installing minimal dependencies (validation only)...$(NC)"
	$(PIP) install pyyaml
	@echo "$(GREEN)✅ Minimal installation complete!$(NC)"
	@echo ""
	@echo "This allows running: make validate"
	@echo "For training, run: make install"

install-gpu:
	@echo "$(BLUE)Installing dependencies with GPU support (CUDA 11.8)...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	$(PIP) install -r requirements.txt --no-deps
	$(PIP) install numpy pandas scipy scikit-learn xgboost optuna pyyaml tqdm matplotlib seaborn
	@echo "$(GREEN)✅ GPU installation complete!$(NC)"
	@echo ""
	@echo "To verify GPU, run: python -c 'import torch; print(torch.cuda.is_available())'"

# ============================================================================
# Testing & Validation
# ============================================================================

test:
	@echo "$(BLUE)Running CLI smoke tests...$(NC)"
	@echo ""
	$(PYTHON) tests/smoke_test_cli.py
	@echo ""

validate:
	@echo "$(BLUE)Running system health check...$(NC)"
	@echo ""
	$(PYTHON) utils/system_check.py
	@echo ""

# ============================================================================
# Training Commands
# ============================================================================

train:
	@echo "$(BLUE)Running example training: MLP model$(NC)"
	@echo ""
	$(PYTHON) run_unified_train.py --model mlp
	@echo ""

train-xgb:
	@echo "$(BLUE)Running XGBoost training$(NC)"
	@echo ""
	$(PYTHON) run_unified_train.py --model xgboost
	@echo ""

train-transformer:
	@echo "$(BLUE)Running Transformer training$(NC)"
	@echo ""
	$(PYTHON) run_unified_train.py --model transformer
	@echo ""

optuna:
	@echo "$(BLUE)Running Optuna hyperparameter optimization (5 trials)$(NC)"
	@echo ""
	cd optuna && $(PYTHON) run_optuna.py --model mlp --n_trials 5
	@echo ""

optuna-full:
	@echo "$(BLUE)Running full Optuna optimization (50 trials)$(NC)"
	@echo ""
	cd optuna && $(PYTHON) run_optuna.py --model mlp --n_trials 50
	@echo ""

seeds:
	@echo "$(BLUE)Running multi-seed experiment (3 seeds)$(NC)"
	@echo ""
	$(PYTHON) run_multiple_seeds.py --model mlp --seeds 3
	@echo ""

seeds-5:
	@echo "$(BLUE)Running multi-seed experiment (5 seeds)$(NC)"
	@echo ""
	$(PYTHON) run_multiple_seeds.py --model mlp --seeds 5
	@echo ""

# ============================================================================
# Maintenance
# ============================================================================

clean:
	@echo "$(BLUE)Cleaning Python cache files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cache cleaned!$(NC)"

clean-all: clean
	@echo "$(BLUE)Deep cleaning (cache + outputs)...$(NC)"
	rm -rf ../../outputs/* 2>/dev/null || true
	rm -rf optuna/*.db 2>/dev/null || true
	@echo "$(GREEN)✅ Deep clean complete!$(NC)"

format:
	@echo "$(BLUE)Formatting code with black...$(NC)"
	@if command -v black >/dev/null 2>&1; then \
		black --line-length 100 . --exclude '/(\.git|__pycache__|\.pytest_cache|\.mypy_cache|venv|\.venv)/'; \
		echo "$(GREEN)✅ Code formatted!$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  black not installed. Install with: pip install black$(NC)"; \
	fi

lint:
	@echo "$(BLUE)Linting code with flake8...$(NC)"
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 . --exclude=__pycache__,.pytest_cache,.mypy_cache,venv,.venv --max-line-length=100; \
		echo "$(GREEN)✅ Linting complete!$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  flake8 not installed. Install with: pip install flake8$(NC)"; \
	fi

# ============================================================================
# Info
# ============================================================================

info:
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║                    System Information                              ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Working Directory: $(shell pwd)"
	@echo ""
	@echo "Package Status:"
	@-$(PYTHON) -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || echo "  PyTorch: Not installed"
	@-$(PYTHON) -c "import numpy; print(f'  NumPy: {numpy.__version__}')" 2>/dev/null || echo "  NumPy: Not installed"
	@-$(PYTHON) -c "import pandas; print(f'  Pandas: {pandas.__version__}')" 2>/dev/null || echo "  Pandas: Not installed"
	@-$(PYTHON) -c "import sklearn; print(f'  Scikit-learn: {sklearn.__version__}')" 2>/dev/null || echo "  Scikit-learn: Not installed"
	@-$(PYTHON) -c "import yaml; print(f'  PyYAML: {yaml.__version__}')" 2>/dev/null || echo "  PyYAML: Not installed"
	@echo ""

# ============================================================================
# Default target
# ============================================================================

.DEFAULT_GOAL := help
