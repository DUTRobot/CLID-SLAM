# Installation Guide for CUDA Neural Points

## Quick Start

### Prerequisites
1. **CUDA Toolkit** (version 11 or higher)
   ```bash
   # Check CUDA installation
   nvcc --version
   ```
2 **PyTorch with CUDA support**
   ```bash
   # Install PyTorch with CUDA
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
3. **C++ Compiler** (GCC 7r Clang5+)
   ```bash
   # Check compiler
   gcc --version
   ```

### Installation Steps

1. **Clone the repository** (if not already done)
   ```bash
   git clone <repository-url>
   cd CLID-SLAM
   ```
2. **Build the CUDA extension**
   ```bash
   cd cuda_neural_points
   chmod +x build.sh
   ./build.sh
   ```3tall the extension**
   ```bash
   pip install -e .
   ```

4. **Test the installation**
   ```bash
   python example_usage.py
   ```

## Detailed Installation

### Step 1: Environment Setup

Create a virtual environment (recommended):
```bash
python -m venv cuda_env
source cuda_env/bin/activate  # On Windows: cuda_env\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118other dependencies
pip install numpy setuptools wheel
```

### Step 3 Build Extension

```bash
cd cuda_neural_points

# Option 1: Use build script
./build.sh

# Option 2: Manual build
python setup.py build_ext --inplace
```

### Step 4: Verify Installation

```python
# Test import
import torch
print(f"PyTorch version: {torch.__version__}")
print(fCUDA available: {torch.cuda.is_available()}")

# Test CUDA extension
try:
    import neural_points_cuda
    print("✓ CUDA extension imported successfully")
except ImportError as e:
    print(f"✗ CUDA extension failed: {e}")
```

## Troubleshooting

### Common Issues

1. **CUDA not found**
   ```bash
   # Install CUDA toolkit
   # Download from NVIDIA website or use package manager
   sudo apt-get install nvidia-cuda-toolkit  # Ubuntu
   ```
2Torch CUDA support missing**
   ```bash
   # Reinstall PyTorch with CUDA
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Compiler errors**
   ```bash
   # Install build tools
   sudo apt-get install build-essential  # Ubuntu
   ```

4mory errors during build**
   ```bash
   # Reduce parallel jobs
   export MAX_JOBS=1
   python setup.py build_ext --inplace
   ```

### Platform-Specific Notes

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit build-essential
```

#### CentOS/RHEL
```bash
sudo yum install cuda-toolkit gcc-c++
```

#### Windows1tall Visual Studio Build Tools
2. Install CUDA Toolkit from NVIDIA website
3. Add CUDA to PATH environment variable

#### macOS
```bash
# Install via Homebrew
brew install cuda
```

### Performance Verification

Run the benchmark to verify performance improvements:

```bash
python example_usage.py
```

Expected output:
```
=== NeuralPoints Performance Benchmark ===
Device: cuda
Feature dimension: 32
Voxel size:0.1-- Original Python Implementation ---
Update time: 0.1234
Query time:0.0567UDA Implementation ---
Update time: 0.0234
Query time: 0.0123s

--- Performance Comparison ---
Update speedup:5.27x
Query speedup: 4.61x
```

## Advanced Configuration

### Custom CUDA Architecture

Edit `setup.py` to target specific GPU architectures:

```python
# In setup.py, modify the CUDA flags
def get_cuda_flags():
    return
       -arch=sm_70',  # For V10
       -arch=sm_80',  # For A10
       -arch=sm_86 For RTX 30 series
    ]
```

### Memory Optimization

For large maps, adjust memory settings:

```python
# In your configuration
config.buffer_size =50000 Larger hash table
config.max_points = 100000More neural points
```

### Debug Mode

Enable debug output:

```python
config.silence = False
```

## Support

If you encounter issues:

1. Check the troubleshooting section above2. Verify your CUDA and PyTorch versions
3. Run the test script to identify specific problems
4. Check GPU memory usage during build
5. Consult the main project documentation

## Uninstallation

To remove the CUDA extension:

```bash
pip uninstall neural_points_cuda
```

The original Python implementation will continue to work normally. 