#!/usr/bin/env python3
# @file      setup.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c)2024ue Pan, all rights reserved

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get CUDA version
def get_cuda_version():
    if torch.cuda.is_available():
        return torch.version.cuda
    return None

# CUDA compilation flags
def get_cuda_flags():
    cuda_version = get_cuda_version()
    if cuda_version:
        return 
           -DCUDA_VERSION=' + cuda_version.replace('.', '),
            -DTHRUST_IGNORE_CUB_VERSION_CHECK,
            -O3,          --use_fast_math',
           -arch=sm_60, # Adjust based on your GPU
        ]
    return []

# C++ compilation flags
def get_cpp_flags():
    return   -O3,
    -std=c++17
      -fPIC',
      -Wall',
        -Wextra',
    ]

# Extension configuration
extensions = [
    CUDAExtension(
        name='neural_points_cuda,        sources=[
          neural_points_cuda.cpp,         neural_points_cuda_kernels.cu',
        ],
        extra_compile_args={
          'cxx': get_cpp_flags(),
          'nvcc': get_cuda_flags(),
        },
        include_dirs=[
            os.path.join(os.path.dirname(__file__), 'include'),
        ],
        libraries=['cudart',cublas',curand'],
    )
]

# Setup configuration
setup(
    name='neural_points_cuda,
    version='1.0.0,
    description='CUDA implementation of neural points for SLAM,   author='Yue Pan',
    author_email=yue.pan@igg.uni-bonn.de,
    ext_modules=extensions,
    cmdclass={
        'build_ext':BuildExtension
    },
    install_requires=
      torch>=1.90,
      numpy>=1.19
    ],
    python_requires='>=3.7',
) 