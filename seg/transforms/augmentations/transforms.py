import cv2
import torch
import random

from functional import *
# from ..builder import TRANSFORMS
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

    def __call__(self, *args, force_apply=False, **kwargs):
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        if (random.random() < self.p) or self.always_apply or force_apply:
            params = self.get_params(**kwargs)
            res = {}
            for key, arg in kwargs.items():
                if arg is not None:
                    target_function = self._get_target_function(key)
                    res[key] = target_function(arg, **params)
            return res
        return kwargs


# @TRANSFORMS.register_module()
class Resize(BaseTransform):
    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, padding=0., prob=1., always_apply=False):
        super(Resize, self).__init__(always_apply, prob)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.padding = padding

    def apply(self, image, **kwargs):
        return padding_resize(image, self.height, self.width, self.interpolation) if kwargs.get('padding') else resize(image, self.height, self.width, self.interpolation)

    def apply_to_mask(self, mask, **kwargs):
        return padding_resize_mask(mask, self.height, self.width, self.interpolation) if kwargs.get('padding') else resize(mask, self.height, self.width, cv2.INTER_NEAREST)

    def get_params(self, **kwargs):
        padding = random.random() < self.padding
        if kwargs.get('image', None):
            return {
                'cols': kwargs['image'].shape[1],
                'rows': kwargs['image'].shape[0],
                'padding': padding,
            }
        elif kwargs.get('images', None):
            return {
                'cols': kwargs['image'][0].shape[1],
                'rows': kwargs['image'][0].shape[0],
                'padding': padding,
            }
        else:
            raise ValueError('No image or images in Resize')

class HorizontalFlip(BaseTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(HorizontalFlip, self).__init__(always_apply, p)

    def apply(self, image):
        return hflip(image)

    def apply_to_mask(self, mask):
        return hflip(mask)

    def get_params(self, **kwargs):
        if kwargs.get('image', None):
            return {
                'cols': kwargs['image'].shape[1],
                'rows': kwargs['image'].shape[0],
            }
        elif kwargs.get('images', None):
            return {
                'cols': kwargs['image'][0].shape[1],
                'rows': kwargs['image'][0].shape[0],
            }
        else:
            raise ValueError('No image or images in Resize')


if __name__ == '__main__':
    ip = '/workspace/mycode/03-seg/seg/local/000.png'
    gp = '/workspace/mycode/03-seg/seg/local/000_mask.png'
    image = cv2.imread(ip)
    mask = cv2.imread(gp)
    # resize_method = Resize(256, 256)
    # resized_image = resize_method.apply(image=image)
    # resized_mask = resize_method.apply_to_mask(mask=mask)
    # cv2.imwrite(ip.replace('.png', '_resized.png'), resized_image)
    # cv2.imwrite(gp.replace('.png', '_resized.png'), resized_mask)


    aug_hflip = HorizontalFlip()
    himage = aug_hflip(image=image)
    hmask = aug_hflip(mask=mask)
    cv2.imwrite(ip.replace('.png', '_hflip.png'), himage)
    cv2.imwrite(gp.replace('.png', '_hflip.png'), hmask)