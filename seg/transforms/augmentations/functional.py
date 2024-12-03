import cv2
import numpy as np
from scipy.ndimage import interpolation

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}


def resize(image, height, width, interpolation=cv2.INTER_LINEAR):
    if image.ndim == 2:
        image_height, image_width = image.shape[:2]
        if height == image_height and width == image_width:
            image = image[:, :, None]
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
        image = image[:, :, None]
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
        mask = mask[:, :, None]
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
    return cv2.flip(image, 1)


def vflip(image):
    return cv2.flip(image, 0)


def cflip(image):
    return cv2.flip(image, -1)


def rotate(image, angle, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None):
    image_height, image_width = image.shape[:2]
    # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
    m = cv2.getRotationMatrix2D((image_width / 2, image_height / 2), angle, 1.0)
    return cv2.warpAffine(image, M=m, dsize=(image_width, image_height), flags=interpolation, borderMode=border_mode,
                          borderValue=value)


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


def adjust_brightness(image, factor):
    if factor == 0:
        return np.zeros_like(image)
    elif factor == 1:
        return image
    if image.dtype == np.uint8:
        lut = np.arange(0, 256) * factor
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        return cv2.LUT(image, lut)
    return clip(image * factor, image.dtype, MAX_VALUES_BY_DTYPE[image.dtype])


def is_grayscale_image(image):
    return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)


def adjust_contrast(image, factor):
    if factor == 1:
        return image

    if is_grayscale_image(image):
        mean = image.mean()
    else:
        mean = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean()

    if factor == 0:
        return np.full_like(image, int(mean + 0.5), dtype=image.dtype)

    if image.dtype == np.uint8:
        lut = np.arange(0, 256) * factor
        lut = lut + mean * (1 - factor)
        lut = clip(lut, image.dtype, 255)
        return cv2.LUT(image, lut)

    return clip(
        image.astype(np.float32) * factor + mean * (1 - factor),
        image.dtype,
        MAX_VALUES_BY_DTYPE[image.dtype]
    )


def adjust_saturation(image, factor, gamma=0):
    if factor == 1:
        return image

    if is_grayscale_image(image):
        gray = image
        return gray
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if factor == 0:
        return gray

    result = cv2.addWeighted(image, factor, gray, 1 - factor, gamma=gamma)
    if image.dtype == np.uint8:
        return result

    # OpenCV does not clip values for float dtype
    return clip(result, image.dtype, MAX_VALUES_BY_DTYPE[image.dtype])


def adjust_hue(image, factor):
    if is_grayscale_image(image) or factor == 0:
        return image

    if image.dtype == np.uint8:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lut = np.arange(0, 256, dtype=np.int16)
        lut = np.mod(lut + 180 * factor, 180).astype(np.uint8)
        image[..., 0] = cv2.LUT(image[..., 0], lut)
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[..., 0] = np.mod(image[..., 0] + factor * 360, 360)
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


def normalize(image, mean, std, scale=1.0):
    #
    if image.ndim == 2:
        mean = mean.mean()
        std = std.mean()
    mean *= scale
    std *= scale

    denominator = np.reciprocal(std, dtype=np.float32)  # 取倒数
    return (image.astype(np.float32) - mean) * denominator


def denormalize(image, mean, std, scale=1.0):
    if image.ndim == 2:
        mean = mean.mean()
        std = std.mean()

    return (image.astype(np.float32) * std + mean) * scale
