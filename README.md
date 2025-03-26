<p align="center">
  <h1 align="center">⚔️CLID-SLAM: A Coupled LiDAR-Inertial Neural Implicit Dense SLAM with Region-Specific SDF Estimation</h1>
  <p align="center">
    <a href="https://github.com/DUTRobot/CLID-SLAM"><img src="https://img.shields.io/github/v/release/PRBonn/PIN_SLAM?label=version" /></a>
    <a href="https://github.com/DUTRobot/CLID-SLAM"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="https://github.com/DUTRobot/CLID-SLAM"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://ieeexplore.ieee.org/abstract/document/10884955"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a>
    <a href="https://github.com/PRBonn/PIN_SLAM/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
  </p>
</p>

## TODO 📝

- [x] Release the source code
- [ ] Enhance and update the README file
- [ ] Include the mathematical theory derivations

## Installation

### Platform Requirements
- Ubuntu 20.04
- GPU (tested on RTX 4090)

### Steps
1. **Clone the repository**
    ```bash
    git clone git@github.com:DUTRobot/CLID-SLAM.git
    cd CLID-SLAM
    ```
   
2. **Create Conda Environment**
   ```bash
   conda create -n slam python=3.12
   conda activate slam
   ```

3. **Install PyTorch**
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

4. **Install ROS Dependencies**
   ```bash
   sudo apt install ros-noetic-rosbag ros-noetic-sensor-msgs
   ```

5. **Install Other Dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```

## Data Preparation

### Download ROSbag Files
Download these essential ROSbag datasets:
- [**Newer College Dataset**](https://ori-drs.github.io/newer-college-dataset/)
- [**SubT-MRS Dataset**](https://superodometry.com/iccv23_challenge_LiI)

### Convert to Sequences
1. Edit `./dataset/converter/config/rosbag2dataset.yaml`.
2. Run:
   ```bash
   python3 ./dataset/converter/rosbag2dataset_parallel.py

## Run CLID-SLAM
```bash
python3 slam.py ./config/run_ncd128.yaml
```
## Acknowledgements 🙏

This project is built upon the open-source project [**PIN-SLAM**](https://github.com/PRBonn/PIN_SLAM), developed by [**PRBonn/YuePanEdward**](https://github.com/YuePanEdward). A huge thanks to the contributors of **PIN-SLAM** for their outstanding work and dedication!
