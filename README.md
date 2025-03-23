# CLID-SLAM: A Coupled LiDAR-Inertial Neural Implicit Dense SLAM with Region-Specific SDF Estimation

For more details, please refer to the [paper](https://ieeexplore.ieee.org/abstract/document/10884955).

## TODO 📝

- [x] Release the source code
- [ ] Enhance and update the README file
- [ ] Include the mathematical theory derivations

## Installation

### Platform requirement
* Ubuntu OS (tested on 20.04)

* With GPU (recommended) or CPU only (run much slower)

* GPU memory requirement (> 6 GB recommended)

* Windows/MacOS with CPU-only mode


### 1. Set up conda environment

```
conda create --name slam python=3.8
conda activate slam
```

### 2. Install the key requirement PyTorch

```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia 
```

### 3. Install other dependency

```
pip3 install -r requirements.txt
```

## Datasets 📚
These datasets provided invaluable resources for the success of this project：
- [**Newer College Dataset**](https://ori-drs.github.io/newer-college-dataset/)  
- [**SubT-MRS Dataset**](https://superodometry.com/iccv23_challenge_LiI)

## Acknowledgements 🙏

This project is built upon the open-source project [**PIN-SLAM**](https://github.com/PRBonn/PIN_SLAM), developed by [**PRBonn/YuePanEdward**](https://github.com/YuePanEdward). A huge thanks to the contributors of **PIN-SLAM** for their outstanding work and dedication!

