# Creating the content of the README.md file
readme_content = """# Grounded-SLAM

**Grounded-SLAM** is a modular system that integrates visual grounding and simultaneous localization and mapping (SLAM) using cutting-edge object detection and language grounding models. This repository leverages [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for zero-shot object grounding.

## 📦 Features

- 🔍 **Object-Aware SLAM:** Enrich SLAM pipelines with semantic object information.
- 🧠 **Zero-Shot Visual Grounding:** Detect objects in real-time with natural language queries.
- 🗺️ **Integrated Mapping:** Annotate and localize objects in SLAM maps.
- 🧩 **Modular Architecture:** Plug-and-play design for integrating various grounding and SLAM backends.

## 🔧 Setup

### 1. Clone the repository with submodules

```bash
git clone --recurse-submodules https://github.com/your-username/Grounded-SLAM.git
cd Grounded-SLAM

git submodule update --init --recursive

pip install -r requirements.txt

cd GroundingDINO
pip install -r requirements.txt
cd ..

python run_grounded_slam.py --config configs/example.yaml

Grounded-SLAM/
├── GroundingDINO/         # Submodule for zero-shot grounding
├── slam/                  # SLAM wrapper and interfaces
├── configs/               # YAML configs for experiments
├── run_grounded_slam.py   # Main entry point
├── requirements.txt
└── README.md

python tests/test_pipeline.py


git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO

python3 -m venv groundingdino_env
source groundingdino_env/bin/activate

pip install --upgrade pip
pip install --default-timeout=120 scipy supervision addict timm transformers yapf opencv-python pycocotools --no-cache-dir

pip install -e .

cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install

sudo apt install librealsense2-dev librealsense2-utils

realsense-viewer

mkdir weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/0.1.0/groundingdino_swint_ogc.pth -P weights/

python demo/inference_on_a_image.py \
  -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
  -p weights/groundingdino_swint_ogc.pth \
  -i demo/demo.jpg \
  -o outputs/ \
  -t "a dog" \
  --cpu-only


source groundingdino_env/bin/activate
python dino_realsense_stream.py

rs-enumerate-devices

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118





