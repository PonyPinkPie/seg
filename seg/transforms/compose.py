from .builder import build_transform

class Compose:
    def __init__(self, transforms):
        """
        按顺序组合多个图像增强变换。
        参数：transforms (Sequence[dict | callable])：要组合的变换对象或配置字典的序列
        """
        self.transforms = []
        self.transforms_dict = dict()
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_transform(transform)
                self.transforms.append(transform)
                name = transform.__class__.__name__
                self.transforms_dict[name] = transform
            elif callable(transform):
                self.transforms.append(transform)
                name = transform.__class__.__name__
                self.transforms_dict[name] = transform
            else:
                raise TypeError('transform must be callable or dict')

    def __call__(self, **data):
        """调用函数以按顺序应用图像增强变换.
        参数:
            data (dict): 一个图像增强变化后的一个字典格式.

        Returns:
           dict: 序列应用后的字典格式.
        """
        for t in self.transforms:
            data = t(**data)
            if data is None:
                return None
        return data

    def __getitem__(self, key):
        return self.transforms_dict[key]
