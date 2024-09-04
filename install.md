## Installation

### Requirements:

- PyTorch >= 1.0. (We use torch==1.2.0) Installation instructions can be found in <https://pytorch.org/get-started/locally/>.
- torchvision==0.2.1(You must use this version)
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- (optional) OpenCV for the webcam demo

### Step-by-step installation

```
conda create --name PCSA python=3.7
conda activate PCSA

pip install torch==1.2.0 torchvision==0.2.1
pip install -r requirements.txt

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

cd PCSA
python setup.py build develop
```