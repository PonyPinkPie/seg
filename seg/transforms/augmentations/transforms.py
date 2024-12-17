import torch
import random
import numpy as np
import numbers
from .functional import *
from ..builder import TRANSFORMS


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


@TRANSFORMS.register_module()
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
        if 'image' in kwargs.keys():
            return {
                'cols': kwargs['image'].shape[1],
                'rows': kwargs['image'].shape[0],
                'padding': padding,
            }
        elif 'images' in kwargs.keys():
            return {
                'cols': kwargs['images'][0].shape[1],
                'rows': kwargs['images'][0].shape[0],
                'padding': padding,
            }
        else:
            raise ValueError('No image or images in Resize')


@TRANSFORMS.register_module()
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


@TRANSFORMS.register_module()
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


@TRANSFORMS.register_module()
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


@TRANSFORMS.register_module()
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


@TRANSFORMS.register_module()
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


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    def __init__(self,
                 height_ratio=1.,
                 width_ratio=1.,
                 padding=False,
                 crop_height=0,
                 crop_width=0,
                 crop_object=False,
                 crop_object_ratio=1.0,
                 **kwargs):
        super(RandomCrop).__init__(**kwargs)
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio
        self.padding = padding
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.crop_object = crop_object
        self.crop_object_ratio = crop_object_ratio

    def apply(self, image, crop_height, crop_width, h_start=0, w_start=0, **kwargs):
        return random_crop(image, crop_height, crop_width, h_start, w_start) if not self.padding \
            else random_crop_padding(image, crop_height, crop_width, h_start, w_start)

    def apply_to_mask(self, mask, crop_height, crop_width, h_start=0, w_start=0, **kwargs):
        return random_crop(mask, crop_height, crop_width, h_start, w_start) if not self.padding \
            else random_crop_padding(mask, crop_height, crop_width, h_start, w_start)

    def _get_yy_xx(self, params):
        if self.crop_object_ratio > random.random():
            if "mask" in params:
                mask = params["mask"]
            elif "masks" in params and len(params["masks"]) > 0:
                mask = params["masks"][0]
            else:
                image = params["image"] if "image" in params else params["images"][0]
                if len(image.shape) == 3: image = image[..., 0]
                _, mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        else:
            image = params["image"] if "image" in params else params["images"][0]
            if len(image.shape) == 3: image = image[..., 0]
            _, mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

        yy, xx = np.where(mask)
        return yy, xx

    def get_params(self, **kwargs):
        height, width = kwargs['image'].shape[:2] if kwargs.get('image', None) is not None \
            else kwargs.get('images')[0].shape[:2]

        if self.crop_height == 0:
            crop_height = int(height * self.height_ratio)
        else:
            crop_height = self.crop_height
        if self.crop_width == 0:
            crop_width = int(width * self.width_ratio)
        else:
            crop_width = self.crop_width
        no_object = True
        if self.crop_object:
            try:
                yy, xx = self._get_yy_xx(kwargs)
                index = random.randint(0, len(yy) - 1)
                coord = [max(int(xx[index] - crop_width * 0.5), 0), max(int(yy[index] - crop_height * 0.5), 0)]
                w_start = min(coord[0] / (width - crop_width), 1.)
                h_start = min(coord[1] / (height - crop_height), 1.)
                no_object = False
            except:
                pass
        if no_object:
            h_start = random.random()
            w_start = random.random()
        return {
            "h_start": h_start,
            "w_start": w_start,
            "rows": width,
            "cols": height,
            "crop_height": crop_height,
            "crop_width": crop_width
        }


@TRANSFORMS.register_module()
class CenterCrop(RandomCrop):
    def get_params(self, **params):
        height, width = params["image"].shape[:2] if params.get("image", None) is not None \
            else params.get("images")[0].shape[:2]
        if self.crop_height == 0:
            crop_height = int(height * self.height_ratio)
        else:
            crop_height = self.crop_height
        if self.crop_width == 0:
            crop_width = int(width * self.width_ratio)
        else:
            crop_width = self.crop_width

        return {
            "h_start": 0.5,
            "w_start": 0.5,
            "rows": width,
            "cols": height,
            "crop_height": crop_height,
            "crop_width": crop_width
        }


@TRANSFORMS.register_module()
class MultiplicativeNoise(BaseTransform):

    def __init__(self, multiplier=(0.9, 1.1), per_channel=False, **kwargs):
        super(MultiplicativeNoise, self).__init__(**kwargs)

        if isinstance(multiplier, (int, float)):
            self.multiplier = -multiplier, +multiplier
        else:
            self.multiplier = multiplier
        self.per_channel = per_channel

    @property
    def targets(self):
        return {"image": self.apply, "images": self.apply_to_images}  # image only transform

    def apply(self, image, multiplier=np.array([1]), **params):
        return multiply(image, multiplier)

    def get_params(self, **kwargs):
        if self.multiplier[0] == self.multiplier[1]:
            return {"multiplier": np.array([self.multiplier[0]])}

        image = kwargs.get("image", None)
        if image is None:
            image = kwargs.get("images")[0]
        # h, w = image.shape[:2]
        if self.per_channel:
            c = 1 if is_grayscale_image(image) else image.shape[-1]
        else:
            c = 1
        multiplier = np.random.uniform(self.multiplier[0], self.multiplier[1], [c])
        if is_grayscale_image(image):
            multiplier = np.squeeze(multiplier)

        return {"multiplier": multiplier}


@TRANSFORMS.register_module()
class GaussNoise(BaseTransform):
    def __init__(self, var_limit=(10.0, 50.0), mean=0, **kwargs):
        super(GaussNoise, self).__init__(**kwargs)
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if var_limit[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")

            self.var_limit = (0, var_limit)
        else:
            raise TypeError(
                "Expected var_limit type to be one of (int, float, tuple, list), got {}".format(type(var_limit))
            )

        self.mean = mean

    @property
    def targets(self):
        return {"image": self.apply, "images": self.apply_to_images}  # image only transform

    def apply(self, img, gauss=None, **kwargs):
        return gauss_noise(img, gauss=gauss)

    def get_params(self, **kwargs):
        image = kwargs["image"] if kwargs.get("image", None) is not None else kwargs.get("images")[0]

        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5
        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))

        gauss = random_state.normal(self.mean, sigma, image.shape)
        return {"gauss": gauss}


@TRANSFORMS.register_module()
class RandomCutout(BaseTransform):
    """Args:
           prob (float): cutout probability.
           n_holes (int | tuple[int, int]): Number of regions to be dropped.
               If it is given as a list, number of holes will be randomly
               selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
           cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
               shape of dropped regions. It can be `tuple[int, int]` to use a
               fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
               shape from the list.
           cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
               candidate ratio of dropped regions. It can be `tuple[float, float]`
               to use a fixed ratio or `list[tuple[float, float]]` to randomly
               choose ratio from the list. Please note that `cutout_shape`
               and `cutout_ratio` cannot be both given at the same time.
           fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
               of pixel to fill in the dropped regions. Default: (0, 0, 0).
       """

    def __init__(self, n_holes=1, cutout_shape=None, cutout_ratio=(0.1, 0.1), fill_in=0, **kwargs):
        super(RandomCutout, self).__init__(**kwargs)

        assert (cutout_shape is None) ^ (
                    cutout_ratio is None), 'Either cutout_shape or cutout_ratio should be specified.'
        assert (isinstance(cutout_shape, (list, tuple)) or isinstance(cutout_ratio, (list, tuple)))
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        self.n_holes = n_holes
        self.fill_in = fill_in
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]

    def apply(self, img, **kwargs):
        return random_cutout(img, **kwargs)

    def apply_to_mask(self, img, **kwargs):
        return random_cutout(img, **kwargs)

    def get_params(self, **kwargs):
        height, width = kwargs["image"].shape[:2] if kwargs.get("image", None) is not None else kwargs.get("images")[
                                                                                                    0].shape[:2]
        # cutout = True if np.random.rand() < self.prob else False
        n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        for _ in range(n_holes):
            x1 = np.random.randint(0, width)
            y1 = np.random.randint(0, height)
            index = np.random.randint(0, len(self.candidates))
            if not self.with_ratio:
                cutout_w, cutout_h = self.candidates[index]
            else:
                cutout_w = int(self.candidates[index][0] * width)
                cutout_h = int(self.candidates[index][1] * height)

            x2 = np.clip(x1 + cutout_w, 0, width)
            y2 = np.clip(y1 + cutout_h, 0, height)

            return {"x1": x1, "x2": x2, "y1": y1, "y2": y2, "fill_in": self.fill_in}


@TRANSFORMS.register_module()
class Normalize(BaseTransform):
    def __init__(self,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 means=[(0.485, 0.456, 0.406)],
                 stds=[(0.229, 0.224, 0.225)],
                 scale=255.0,
                 **kwargs):
        super(Normalize, self).__init__(**kwargs)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.means = means
        self.stds = stds
        self.scale = scale

    @property
    def targets(self):
        return {'image': self.apply, 'images': self.apply_to_images}

    def get_params(self, **kwargs):
        return {'mean': self.mean, 'std': self.std}

    def apply(self, image, mean, std, **kwargs):
        return normalize(image, mean, std, self.scale)

    def apply_to_images(self, images, **kwargs):
        assert len(images) == len(self.means) == len(self.stds), \
            (f"len of images must be equal means and stds, but got len(images) = {len(images)}, "
             f"len(self.means) = {len(self.means)} len(self.stds) = {len(self.stds)}")
        return [self.apply(image, np.array(mean), np.array(std)) for image, mean, std in
                zip(images, self.means, self.stds)]


@TRANSFORMS.register_module()
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
