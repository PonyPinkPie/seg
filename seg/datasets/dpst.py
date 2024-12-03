from torch.utils.data import Dataset, ConcatDataset

from seg.transforms import Compose


class DpstDataset(Dataset):
    def __init__(self,
                 root=None,
                 mode=None,
                 transforms=None,
                 image_labels=[],
                 shape_labels=[],
                 logger=None,
                 ):
        self.root = root
        self.mode = mode
        self.transform = transforms if transforms is None else Compose(transforms)
        self.image_labels = sorted(image_labels)
        self.shape_labels = sorted(shape_labels)
        self.logger = logger
        self._class_label_dict = self.get_class_label_dict()
        self._label_class_dict = self.get_label_class_dict()
        self.load_data()


    def load_data(self):


        pass

    def get_class_label_dict(self):
        return {name: i for i, name in enumerate(self.shape_labels)}

    def get_label_class_dict(self):
        return {label:cls for label, cls in self._class_label_dict.items()}

    @property
    def class2label(self):
        return self._class_label_dict

    @property
    def label2class(self):
        return self._label_class_dict



