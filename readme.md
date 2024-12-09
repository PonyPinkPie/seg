1. environment
```shell
conda create -n seg python==3.8.5 -y
conda activate seg

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118  -i https://pypi.doubanio.com/simple

pip install -r requirements.txt -i https://pypi.doubanio.com/simple
#pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

```

# TODO
```shell
1、损失函数（看看sigmoid）
2、优化器（学习率放大）
3、模型（推理结果）
4、数据增强（变换是否正确）resize ok, 其他未确认
6、评价指标未开始
7、模型导出

```