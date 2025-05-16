# Visual-SLAM-DOF-w-GDino

**Grounded-SLAM** is a modular system that integrates visual grounding and simultaneous localization and mapping (SLAM) using cutting-edge object detection and language grounding models. This repository leverages [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for zero-shot object grounding.

## Features

- **Object-Aware SLAM:** Enrich SLAM pipelines with semantic object information.
- **Zero-Shot Visual Grounding:** Detect objects in real-time with natural language queries.
- **Integrated Mapping:** Annotate and localize objects in SLAM maps.
- **Modular Architecture:** Plug-and-play design for integrating various grounding and SLAM backends.

---

## Setup: Grounded-SLAM

### 1. Clone the repository with submodules

# Grounded-SLAM: A Framework Integrating Grounding DINO + SAM with ORB-SLAM2 and ORB-SLAM3

This repository provides a modular setup to run dynamic object filtering with Grounding DINO + Segment Anything Model (SAM), and visualize the results with ORB-SLAM2 and ORB-SLAM3 SLAM backends. It supports integration with RealSense cameras and ROS (Noetic) using Pangolin viewer.

---

## Repository Structure
dino_ws/
├── src/
│ ├── gdino_filter/ # ROS package using Grounding DINO + SAM
│ ├── ORB_SLAM2_NOETIC/ # ROS-compatible fork of ORB-SLAM2
│ ├── orbslam3_ros/ # ROS wrapper for ORB-SLAM3 (custom)
├── vision_models/ # Contains Grounding DINO and SAM model codebases

## System Requirements

- Ubuntu 20.04 or later
- ROS Noetic
- Python 3.10 (for Grounding DINO + SAM)
- Python 3.8 (for ROS nodes)
- CUDA-compatible GPU + NVIDIA drivers
- RealSense D435i (or compatible RGB-D camera)
  
### 2. Python environment setup

```bash
python3 -m venv groundingdino_env
source groundingdino_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

cd GroundingDINO
pip install -r requirements.txt
pip install -e .
cd ..
```

### 3. RealSense SDK installation
```bash
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install

sudo apt install librealsense2-dev librealsense2-utils
realsense-viewer
```

### 4. GroundingDINO model weights
```bash
mkdir weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/0.1.0/groundingdino_swint_ogc.pth -P weights/
```

### 5. Run test script
```bash
python run_grounded_slam.py --config configs/example.yaml
python tests/test_pipeline.py
```
### 6. Try demo with image input
```bash
python demo/inference_on_a_image.py \
  -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
  -p weights/groundingdino_swint_ogc.pth \
  -i demo/demo.jpg \
  -o outputs/ \
  -t "a dog" \
```

## Setup: ORB-SLAM3 with RealSense D435i (Stereo-Inertial)

### 1. Clone ORB-SLAM3
```bash
cd ~/slam_ws/src
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3
```
### 2. Install Pangolin (compatible version)
```bash
sudo apt install libglew-dev libboost-dev libboost-thread-dev libboost-filesystem-dev
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

# Download OpenCV 4.5.5 source
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.5.zip
unzip opencv.zip && cd opencv-4.5.5
mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)
sudo make install

```
### 3. Build ORB-SLAM2/3

```bash
cd ~/dino_ws/src/ORB_SLAM2_NOETIC
chmod +x build.sh
./build.sh
cd ~/dino_ws
catkin_make
```

```bash
cd ~/dino_ws/src/orbslam3_ros/ORB_SLAM3
chmod +x build.sh
./build.sh
cd ~/dino_ws
catkin_make
```
### 4. Run Stereo-Inertial D435i example
```bash
cd ~/slam_ws/src/ORB_SLAM3
./Examples_old/Stereo-Inertial/stereo_inertial_realsense_D435i_old Vocabulary/ORBvoc.txt Examples/Stereo-Inertial/RealSense_D435i.yaml
```

### 5. Fix permission issues for IMU (if needed)
```bash
sudo chmod a+r /sys/bus/iio/devices/iio:device*/scan_elements/*
sudo chmod a+r /dev/iio:device*

Or add a udev rule:

sudo nano /etc/udev/rules.d/99-realsense-libusb.rules

Add the line:

SUBSYSTEM=="iio", GROUP="plugdev", MODE="0666"

sudo udevadm control --reload-rules && sudo udevadm trigger

```
### Directory Structure
```bash
Grounded-SLAM/
├── GroundingDINO/               # Submodule for zero-shot grounding
├── slam/                        # SLAM wrapper and interfaces
├── configs/                     # YAML configs for experiments
├── run_grounded_slam.py         # Main entry point
├── requirements.txt
├── README.md
├── weights/
│   └── groundingdino_swint_ogc.pth
├── demo/
│   └── inference_on_a_image.py
└── tests/
    └── test_pipeline.py

slam_ws/
└── src/
    └── ORB_SLAM3/
        ├── Vocabulary/
        │   └── ORBvoc.txt
        └── Examples/
            └── Stereo-Inertial/
                └── RealSense_D435i.yaml
```
### ORB SLAM WITHOUT DINO FILTER
![paach_15_OS2](https://github.com/user-attachments/assets/f5215ce9-451f-4f21-8254-f2e76264c7e0)


### ORB SLAM WITH DINO FILTER
![paach_15_OS2_DOF](https://github.com/user-attachments/assets/f1e8c7b4-fd02-46bf-9a30-578ad57ee733)


### Refrences
GroundingDINO

ORB-SLAM3

Intel RealSense SDK







  

