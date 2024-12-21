# 1. environment
```shell
conda create -n seg python==3.8.5 -y
conda activate seg

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118  -i https://pypi.doubanio.com/simple

pip install -r requirements.txt -i https://pypi.doubanio.com/simple
#pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

```


# 2. prepare data
open `config/labelme.json`, 
- common
  - replace `workdir` with `your own workdir`
- data
  - model
    - replace `pretrained` with `your own pretrained path`
  - train and valid
    - replace `root` with `your own data root`
    - replace `shape_labels` with `your own shape_labels`
- 
```shell
python tools/train.py --cfg ./config/labelme.json
```