#!/bin/bash

# Build script for CUDA Neural Points Extension
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c)2024ue Pan, all rights reserved

set -e

echo "Building CUDA Neural Points Extension...# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA toolkit.exit 1Check CUDA version
CUDA_VERSION=$(nvcc --version | grep release| awk {print $6}| cut -c2-)
echo CUDA version: $CUDA_VERSION"

# Check if PyTorch is installed
if ! python -cimport torch" &> /dev/null; then
    echo "Error: PyTorch not found. Please install PyTorch first.exit 1
fi

# Check PyTorch CUDA support
if python -c "import torch; print(CUDAavailable:', torch.cuda.is_available())" | grep -q "True; then
    echo "PyTorch CUDA support: Available"
else
    echoWarning: PyTorch CUDA support not available. Building CPU-only version."
fi

# Create build directory
mkdir -p build
cd build

# Build the extension
echo "Building extension..."
python ../setup.py build_ext --inplace

# Test the build
echo "Testing build..."
python -c
try:
    import neural_points_cuda
    print('✓ CUDA extension built successfully')
except ImportError as e:
    print(f'✗ CUDA extension failed to import: {e})   exit(1)
"

echo "Build completed successfully!
echo 
echo "To install the extension, run:"
echopip install -e .
echo ""
echo "To test the extension, run:
echopython ../example_usage.py" 