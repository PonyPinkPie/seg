from abc import ABCMeta, abstractmethod
from collections import defaultdict
from seg.transforms.compose import Compose
from torch.utils.data import Dataset, ConcatDataset
from os.path import exists as ope
from seg.utils.io import ls_folder, IMAGE_POSTFIX

class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 root,
                 mode,
                 transforms,
                 image_labels=[],
                 shape_labels=[],
                 logger=None,
                 **kwargs):
        assert ope(root), f"Dataset root={root} does not exist."
        self.root = root
        self.mode = mode
        self.transforms = transforms if transforms is None else Compose(transforms)
        self.image_labels = image_labels
        self.shape_labels = shape_labels
        self.image_list = []
        self.label_list = []
        self.load_data()
        self.logger = logger
        self._class_label_dict = self.get_class_label_dict()
        self._build_label_index_dict()

    def load_data(self):
        raise NotImplementedError

    def get_class_label_dict(self):
        return {n:i for i, n in enumerate(self.image_labels)}

    def _build_label_index_dict(self):
        """
        记录每个类别对应的index索引，方便一些采样方法
        """
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index = label.get("label", 1)
            index_dict[index].append(i)
        self.index_dict = index_dict

    def label_index_dict(self):
        return self.index_dict