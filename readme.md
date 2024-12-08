1. environment
```shell
conda create -n seg python==3.8.5 -y
conda activate seg

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118  -i https://pypi.doubanio.com/simple

pip install -r requirements.txt -i https://pypi.doubanio.com/simple
#pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

```
accimage==0.2.0
ADEval==1.1.0
apex==0.9.10dev
einops==0.8.0
faiss_gpu==1.7.2
FrEIA==0.2
FrEIA==0.2
fvcore==0.1.5.post20221221
geomloss==0.2.6
imgaug==0.4.0
mamba_ssm==2.2.2
matplotlib==3.7.5
mmdet==2.25.3
numba==0.58.1
numpy==1.24.1
numpy_hilbert_curve==1.0.1
opencv_python==4.10.0.84
pandas==2.0.3
Pillow==11.0.0
pycocotools==2.0.7
pyzorder==0.0.2
scikit_learn==1.3.2
scipy==1.14.1
skimage==0.0
tabulate==0.9.0
tensorboardX==2.6.2.2
tensorboardX==2.6.2.2
timm==0.8.15.dev0
torch==2.1.2+cu118
torchvision==0.16.2+cu118
tqdm==4.66.5
