# =============================================================================
# Python ML Study Guide - Docker Container
# =============================================================================
# Purpose: Reproducible ML development environment with virtual environment
# Base: Python 3.11 slim for optimal size and performance
# =============================================================================

FROM python:3.11-slim AS base

# -----------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files
# PYTHONUNBUFFERED: Ensures output is sent straight to terminal (useful for logs)
# PYTHONFAULTHANDLER: Enable fault handler for better error reporting
# PIP_NO_CACHE_DIR: Don't cache pip downloads to save space
# PIP_DISABLE_PIP_VERSION_CHECK: Skip pip version check for faster installs
# -----------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/app/venv \
    PATH="/app/venv/bin:$PATH"

# -----------------------------------------------------------------------------
# System Dependencies
# -----------------------------------------------------------------------------
# Install essential system packages for ML libraries
# - build-essential: Compiler tools for C extensions
# - git: Version control
# - curl: Download utilities
# - libgomp1: OpenMP support for parallel processing
# - libglib2.0-0: Required by OpenCV
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# -----------------------------------------------------------------------------
# Create Application Directory and Virtual Environment
# -----------------------------------------------------------------------------
WORKDIR /app

# Create virtual environment inside the container
RUN python -m venv $VIRTUAL_ENV

# Upgrade pip and install wheel for faster package installs
RUN pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Install Python Dependencies
# -----------------------------------------------------------------------------
# Copy requirements first for better Docker layer caching
# This means dependencies are only reinstalled when requirements.txt changes
COPY requirements.txt .

# Install all ML dependencies
RUN pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Copy Application Code
# -----------------------------------------------------------------------------
# Copy source code last to maximize cache utilization
COPY . .

# -----------------------------------------------------------------------------
# Create Non-Root User for Security
# -----------------------------------------------------------------------------
# Running as non-root is a security best practice
RUN useradd --create-home --shell /bin/bash mluser \
    && chown -R mluser:mluser /app

USER mluser

# -----------------------------------------------------------------------------
# Default Command
# -----------------------------------------------------------------------------
# Start Jupyter Lab by default for interactive development
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
