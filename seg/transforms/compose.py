from .builder import build_transform


class Compose(object):
    def __init__(self, transforms):
        self.transforms = []
        self.transforms_dict = transforms
        for t in transforms:
            if isinstance(t, dict):
                transform = build_transform(t)
                self.transforms.append(transform)
                name = transform.__class__.__name__
                self.transforms_dict[name] = transform
