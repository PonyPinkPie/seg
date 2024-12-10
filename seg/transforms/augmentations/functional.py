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

def get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start):
    y1 = max(int((height-crop_height)*h_start), 0)
    y2 = y1+ crop_height
    x1 = max(int((width-crop_width)* w_start), 0)
    x2 = x1+ crop_width
    return x1, y1, x2, y2

def random_crop(image, crop_height, crop_width, h_start, w_start):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)
    image = image[y1:y2, x1:x2]
    return image

def random_crop_padding(image, crop_height, crop_width, h_start, w_start, border_type=cv2.BORDER_CONSTANT, value=0):
    height, width = image.shape[:2]
    y1 = max(int((height-crop_height)*h_start), 0)
    y2 = y1+ crop_height
    x1 = max(int((width-crop_width)* w_start), 0)
    x2 = x1+ crop_width
    crop_img = image[y1:y2, x1:x2]
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        left_padding = -x1 if x1 < 0 else 0
        right_padding = x2-width if x2 > width else 0
        top_padding = -y1 if y1 < 0 else 0
        bottom_padding = y2-height if y2 > height else 0
        crop_img = cv2.copyMakeBorder(crop_img, top_padding, bottom_padding, left_padding, right_padding,
                                        borderType=border_type, value=value)
    return crop_img

def _multiply_non_uint8(img, multiplier):
    dtype = img.dtype
    maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
    return  clip(img * multiplier,  dtype, maxval)


def _multiply_uint8(img, multiplier):
    dtype = img.dtype
    maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
    img = img.astype(np.float32)
    return clip(np.multiply(img, multiplier), dtype, maxval)

def _multiply_uint8_optimized(img, multiplier):
    if is_grayscale_image(img) or len(multiplier) == 1:
        multiplier = multiplier[0]
        lut = np.arange(0, 256, dtype=np.float32)
        lut *= multiplier
        lut = clip(lut, np.uint8, MAX_VALUES_BY_DTYPE[img.dtype])
        func = _maybe_process_in_chunks(cv2.LUT, lut=lut)
        return func(img)

    channels = img.shape[-1]
    lut = [np.arange(0, 256, dtype=np.float32)] * channels
    lut = np.stack(lut, axis=-1)

    lut *= multiplier
    lut = clip(lut, np.uint8, MAX_VALUES_BY_DTYPE[img.dtype])

    images = []
    for i in range(channels):
        func = _maybe_process_in_chunks(cv2.LUT, lut=lut[:, i])
        images.append(func(img[:, :, i]))
    return np.stack(images, axis=-1)

def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.

    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.

    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.

    Returns:
        numpy.ndarray: Transformed image.

    """

    def __process_fn(img):
        num_channels = img.shape[2] if len(img.shape) == 3 else 1
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn


def multiply(img, multiplier):
    """
    Args:
        img (numpy.ndarray): Image.
        multiplier (numpy.ndarray): Multiplier coefficient.

    Returns:
        numpy.ndarray: Image multiplied by `multiplier` coefficient.

    """
    if img.dtype == np.uint8:
        if len(multiplier.shape) == 1:
            return _multiply_uint8_optimized(img, multiplier)
        return _multiply_uint8(img, multiplier)
    return _multiply_non_uint8(img, multiplier)

def gauss_noise(image, gauss):
    dtype = image.dtype
    image = image.astype("float32")
    if len(image.shape) != len(gauss.shape):
        if len(image.shape) == 3: # if img is rgb
            gauss = np.expand_dims(gauss, axis=-1)
        else:
            # img is grey
            gauss = np.mean(gauss, axis=-1)
    image = image + gauss
    maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
    return np.clip(image, 0, maxval).astype(dtype)

def preserve_channel_dim(func):
    """
    Preserve dummy channel dim.

    """
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        if len(shape) == 2:
            img = np.expand_dims(img, axis=-1)
            result = func(img, *args, **kwargs)
            result = np.squeeze(result)
        else:
            result = func(img, *args, **kwargs)
        return result

    return wrapped_function

@preserve_channel_dim
def random_cutout(img, x1, y1, x2, y2, fill_in):
    img[y1:y2, x1:x2, :] = fill_in
    return img

@preserve_channel_dim
def normalize(image, mean, std, scale=1.0):
    #
    if image.ndim == 2:
        mean = mean.mean()
        std = std.mean()
    if image.max() > 1:
        image = image.astype(np.float32) / 255
    if mean.max() > 1 and std.max() > 1:
        mean = mean.astype(np.float32) / 255
        std = std.astype(np.float32) / 255
    denominator = np.reciprocal(std, dtype=np.float32)  # 取倒数
    return (image.astype(np.float32) - mean) * denominator


# def denormalize(image, mean, std, scale=1.0):
#     if image.ndim == 2:
#         mean = mean.mean()
#         std = std.mean()
#     if image.max() < 1:
#         image = image * 255
#     if mean.max() < 1 and std.max() < 1:
#         mean = mean * 255
#         std = std * 255
#     return (image.astype(np.float32) * std + mean).astype(np.uint8)
