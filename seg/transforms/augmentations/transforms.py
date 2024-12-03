import torch

from functional import *
# from ..builder import TRANSFORMS
import random
import numpy as np
import numbers


class BaseTransform:
    def __init__(self, always_apply=False, p=0.5, **kwargs):
        self.always_apply = always_apply
        self.p = p

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "images": self.apply_to_images,
            "masks": self.apply_to_masks,
            "boxes": self.apply_to_bboxes
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
        return padding_resize(image, self.height, self.width, self.interpolation) if kwargs.get('padding') else resize(
            image, self.height, self.width, self.interpolation)

    def apply_to_mask(self, mask, **kwargs):
        return padding_resize_mask(mask, self.height, self.width, self.interpolation) if kwargs.get(
            'padding') else resize(mask, self.height, self.width, cv2.INTER_NEAREST)

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
            raise ValueError('No image or images in HorizontalFlip')


class VerticalFlip(BaseTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(VerticalFlip, self).__init__(always_apply, p)

    def apply(self, image):
        return vflip(image)

    def apply_to_mask(self, mask):
        return vflip(mask)

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
            raise ValueError('No image or images in VerticalFlip')


class CenterFlip(BaseTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(CenterFlip, self).__init__(always_apply, p)

    def apply(self, image):
        return cflip(image)

    def apply_to_mask(self, mask):
        return cflip(mask)

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
            raise ValueError('No image or images in CenterFlip')


class Rotate(BaseTransform):
    def __init__(self,
                 always_apply=False,
                 p=0.5,
                 limit=[-5, 5],
                 interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101,
                 value=0,
                 mask_value=0,
                 ):
        super(Rotate, self).__init__(always_apply, p)
        if isinstance(limit, (float, int)):
            self.limit = [-abs(limit), abs(limit)]
        else:
            self.limit = limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, image, angle=0, interpolation=cv2.INTER_LINEAR, **kwargs):
        return rotate(image, angle, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, mask, angle=0, **kwargs):
        return rotate(mask, angle, cv2.INTER_NEAREST, self.border_mode, self.mask_value)

    def get_params(self, **kwargs):
        if kwargs.get('image', None):
            return {
                'cols': kwargs['image'].shape[1],
                'rows': kwargs['image'].shape[0],
                'angle': random.uniform(*self.limit)
            }
        elif kwargs.get('images', None):
            return {
                'cols': kwargs['image'][0].shape[1],
                'rows': kwargs['image'][0].shape[0],
                'angle': random.uniform(*self.limit)
            }
        else:
            raise ValueError('No image or images in Rotate')


class ColorJitter(BaseTransform):
    def __init__(self,
                 brightness=0.2,
                 contrast=0.2,
                 saturation=0.2,
                 hue=0.2,
                 **kwargs):
        super(ColorJitter, self).__init__(**kwargs)
        self.brightness = self.__check_values(brightness, "brightness")
        self.contrast = self.__check_values(contrast, "contrast")
        self.saturation = self.__check_values(saturation, "saturation")
        self.hue = self.__check_values(hue, "hue", offset=0, bounds=[-0.5, 0.5], clip=False)

    @property
    def targets(self):
        return {'image': self.apply, 'images': self.apply_to_images}

    def __check_values(self, value, name, offset=1, bounds=(0, float("inf")), clip=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be positive")

            value = [offset - value, offset + value]

            if clip:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
                raise ValueError("{} values should be between {}".format(name, bounds))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))
        return value

    def apply(self, image, transform=(), **kwargs):
        for t in transform:
            image = t(image)
        return image

    def get_params(self, **kwargs):
        brightness = random.uniform(*self.brightness)
        contrast = random.uniform(*self.contrast)
        saturation = random.uniform(*self.saturation)
        hue = random.uniform(*self.hue)

        transforms = [
            lambda x: adjust_brightness(x, brightness),
            lambda x: adjust_contrast(x, contrast),
            lambda x: adjust_saturation(x, saturation),
            lambda x: adjust_hue(x, hue),
        ]
        random.shuffle(transforms)

        return {"transforms": transforms}


class Normalize(BaseTransform):
    def __init__(self,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 means=[(0.485, 0.456, 0.406)],
                 stds=[(0.229, 0.224, 0.225)],
                 scale=255.0,
                 **kwargs):
        super(Normalize, self).__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.means = means
        self.stds = stds
        self.scale = scale

    @property
    def targets(self):
        return {'image': self.apply, 'images': self.apply_to_images}

    def apply(self, image, mean, std, **kwargs):
        return normalize(image, mean, std, self.scale)

    def apply_to_images(self, images, **kwargs):
        assert len(images) == len(self.means) == len(self.stds), \
            f"len of images must be equal means and stds, but got len(images) = {len(images)}, len(self.means) = {len(self.means)} len(self.stds) = {len(self.stds)}"
        return [self.apply(image, np.array(mean), np.array(std)) for image, mean, std in
                zip(images, self.means, self.stds)]


class ToTensor(BaseTransform):
    def __init__(self, transpose_mask=False, **kwargs):
        super(ToTensor, self).__init__(**kwargs)
        self.transpose_mask = transpose_mask

    def apply(self, image, **kwargs):
        if len(image.shape) not in [2, 3]:
            raise ValueError(f"ToTensor only supports images in HW or HWC format")

        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)
        return torch.from_numpy(image.transpose(2, 0, 1))

    def apply_to_images(self, images, **kwargs):
        images = super(ToTensor, self).apply_to_images(images, **kwargs)
        images = torch.cat(images, dim=0).float()
        return images

    def apply_to_mask(self, mask, **kwargs):
        if self.transpose_mask and mask.dim() == 3:
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(mask).float()


if __name__ == '__main__':
    ip = '/workspace/mycode/03-seg/seg/local/000.png'
    gp = '/workspace/mycode/03-seg/seg/local/000_mask.png'
    image = cv2.imread(ip)
    mask = cv2.imread(gp)
    # resize_method = Resize(256, 256)
    # resized_image = resize_method.apply(image=image)
    # resized_mask = resize_method.apply_to_mask(mask=mask)
    # cv2.imwrite(ip.replace('000.png', 'resize_000.png'), resized_image)
    # cv2.imwrite(gp.replace('000_mask.png', 'resize_000_mask.png'), resized_mask)

    # aug_hflip = HorizontalFlip()
    # himage = aug_hflip.apply(image=image)
    # hmask = aug_hflip.apply_to_mask(mask=mask)
    # cv2.imwrite(ip.replace('000.png', 'hflip_000.png'), himage)
    # cv2.imwrite(gp.replace('000_mask.png', 'hflip_000_mask.png'), hmask)
    #
    # aug_vflip = VerticalFlip()
    # vimage = aug_vflip.apply(image=image)
    # vmask = aug_vflip.apply_to_mask(mask=mask)
    # cv2.imwrite(ip.replace('000.png', 'vflip_000.png'), vimage)
    # cv2.imwrite(gp.replace('000_mask.png', 'vflip_000_mask.png'), vmask)

    # aug_cflip = CenterFlip()
    # cimage = aug_cflip.apply(image=image)
    # cmask = aug_cflip.apply_to_mask(mask=mask)
    # cv2.imwrite(ip.replace('000.png', 'cflip_000.png'), cimage)
    # cv2.imwrite(gp.replace('000_mask.png', 'cflip_000_mask.png'), cmask)

    aug_cj = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    cjimage = aug_cj.apply(image=image, angle=5)
    cv2.imwrite(ip.replace('000.png', 'cj_000.png'), cjimage)
