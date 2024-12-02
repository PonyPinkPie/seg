import random

class BaseTransform:
    def __init__(self, always_apply=False, p=0.5, **kwargs):
        self.always_apply = always_apply
        self.p = p

    @property
    def targets(self):
        return {
            "image":self.apply,
            "mask":self.apply_to_mask,
            "images":self.apply_to_images,
            "masks":self.apply_to_masks,
            "boxes":self.apply_to_bboxes
        }

    def get_params(self, **kwargs):
        return {}

    def _get_target_function(self, k):
        return self.targets.get(k, lambda x, **p: x)

    def apply(self, image, **kwargs):
        raise NotImplementedError

    def apply_to_mask(self, mask, **kwargs):
        return self.apply(mask, **kwargs)

    def apply_to_images(self, images, **kwargs):
        return [self.apply(img, **kwargs) for img in images]

    def apply_to_masks(self, masks, **kwargs):
        return [self.apply_to_mask(mask, **kwargs) for mask in masks]

    def apply_to_bbox(self, bbox, **kwargs):
        raise NotImplementedError("Method apply_to_bbox is not implemented in class " + self.__class__.__name__)

    def apply_to_bboxes(self, bboxes, **kwargs):
        return [self.apply_to_bbox(bbox[:4], **kwargs) for bbox in bboxes]

    def __call__(self, *args, force_apply, **kwargs):
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        if random.random() < self.p or self.always_apply or force_apply:
            params = self.get_params(**kwargs)
            res = {}
            for k, v in kwargs.items():
                if v is not None:
                    target_function = self._get_target_function(k)
                    res[k] = target_function(v, **params)
            return res
        return kwargs



