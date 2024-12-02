import cv2
import numpy as np
from scipy.ndimage import interpolation


def resize(image, height, width, interpolation=cv2.INTER_LINEAR):
    if image.ndim == 2:
        image_height, image_width = image.shape[:2]
        if height == image_height and width == image_width:
            image =  image[:,:, None]
    elif image.ndim == 3:
        image_height, image_width, c = image.shape
        if height == image_height and width == image_width:
            return image
    else:
        raise ValueError('image must be 2 or 3 dimensional')

    image = cv2.resize(image, (width, height), interpolation=interpolation)
    return image

def padding_resize(image, height, width, interpolation=cv2.INTER_LINEAR):
    if image.ndim == 2:
        image_height, image_width = image.shape
        image = image[:,:, None]
    elif image.ndim == 3:
        image_height, image_width, c = image.shape
    else:
        raise ValueError(f'Unsupported image shape: {image.shape}')

    resize_ratio = min(height / image_height, width / image_width)
    new_height = int(resize_ratio * image_height)
    new_width = int(resize_ratio * image_width)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    padding_image = np.zeros_like(image, dtype=resized_image.dtype)

    min_h = (image_height - new_height) // 2
    max_h = min_h + new_height
    min_w = (image_width - new_width) // 2
    max_w = min_w + new_width

    padding_image[min_h:max_h, min_w:max_w, ...] = resized_image
    return padding_image

def padding_resize_mask(mask, height, width):
    if mask.ndim == 2:
        mask_height, mask_width = mask.shape
        mask = mask[:,:, None]
    elif len(mask.shape) == 3:
        mask_height, mask_width, c = mask.shape
    else:
        raise ValueError(f'Unsupported mask shape: {mask.shape}')

    mask_ratio = min(height / mask_height, width / mask_width)
    new_height = int(mask_ratio * mask_height)
    new_width = int(mask_ratio * mask_width)
    resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    padding_mask = np.zeros_like(resized_mask, dtype=resized_mask.dtype)

    min_h = (mask_height - new_height) // 2
    max_h = min_h + new_height
    min_w = (mask_width - new_width) // 2
    max_w = min_w + new_width
    padding_mask[min_h:max_h, min_w:max_w, ...] = resized_mask
    return padding_mask


def hflip(image):
    return cv2.flip(image, -1)








































