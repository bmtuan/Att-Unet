import os

import numpy as np
from random import randint

import cv2
from PIL import Image, ImageOps

from transform import change_transform_origin
from params import STD, MEAN


def resize_crop_padding(image_path, label_path, temp_dir):
    basename = os.path.basename(image_path).split(".")[0]
    temp_image_path = os.path.join(temp_dir, "%s.png" % basename)
    temp_mask_path = os.path.join(temp_dir, "%s_mask.png" % basename)
    temp_label_path = os.path.join(temp_dir, "%s.npy" % basename)
    # print(temp_image_path)
    # print(temp_mask_path)
    # print(temp_label_path)
    # if os.path.exists(temp_image_path) and os.path.exists(temp_label_path):
    #     image = Image.open(temp_image_path).convert('RGB')
    #     image = np.array(image)
    #     print('1231313123')
    #     mask = np.load(temp_label_path)
    #     mask = mask > 200
    #     mask = np.array(mask, dtype=np.uint8)
    #     # print('aaa', np.unique(mask))
    # else:
    # loading data
    image = Image.open(image_path).convert('RGB')
    # print(bool(image))
    image = ImageOps.exif_transpose(image)
    image = np.array(image)
    # image = cv2.imread(image_path)

    # mask = np.load(label_path)
    # print(label_path)
    mask = cv2.imread(label_path)

    assert image.shape == mask.shape
    # print(mask.shape, image.shape)
    target_dim = (512, 288)  # width x height

    scale_factor = float(target_dim[0]) / image.shape[1]

    # resize to fit target width
    target_1 = (target_dim[0], int(image.shape[0] * scale_factor))
    image = cv2.resize(image, target_1, interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, target_1, interpolation=cv2.INTER_NEAREST)
    # padding
    # print('-----')
    # print(mask.shape, image.shape)
    if image.shape[0] < target_dim[1]:
        pad_size = abs(target_dim[1] - image.shape[0]) // 2
        image = cv2.copyMakeBorder(
            image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
        mask = cv2.copyMakeBorder(
            mask, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, 0)
    else:
        crop_size = abs(target_dim[1] - image.shape[0]) // 2
        shape_0 = image.shape[0]
        image = image[crop_size: shape_0 - crop_size, ...]
        mask = mask[crop_size: shape_0 - crop_size, ...]
    #     print(crop_size, image.shape, mask.shape)
    # print(mask.shape)
    # print(target_dim)
    image = cv2.resize(image, target_dim, interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, target_dim, interpolation=cv2.INTER_NEAREST)

    # image = cv2.resize(image, target_dim, interpolation=cv2.INTER_NEAREST)
    # mask  = cv2.resize(mask, target_dim, interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(temp_image_path, image)
    cv2.imwrite(temp_mask_path, mask * 255)
    np.save(temp_label_path, mask)

    return image, mask


def val_resize_crop_padding(image_path):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    target_dim = (512, 288)  # width x height

    scale_factor = float(target_dim[0]) / image.shape[1]
    # resize to fit target width
    image = cv2.resize(image, (target_dim[0], int(
        image.shape[0] * scale_factor)), interpolation=cv2.INTER_NEAREST)
    if image.shape[0] < target_dim[1]:
        pad_size = abs(target_dim[1] - image.shape[0]) // 2
        image = cv2.copyMakeBorder(
            image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    image = cv2.resize(image, target_dim, interpolation=cv2.INTER_NEAREST)

    return image


def normalize_image(x):
    """ Normalize the image based on training mean and std.
    Args
        x: np.array of shape (None, None, depth)
    Returns
        The data normalized
    """
    # covert always to float32 to keep compatibility with opencv
    x = x.astype(np.float32)
    # h, w, d = x.shape
    # for i in range(d):
    # 	x[..., i] -= MEAN[i]
    # 	x[..., i] /= STD[i]
    x = x / 255.0

    return x


class TransformParameters:
    """ Struct holding parameters determining how to apply a transformation to an image.

    Args
        fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval:                  Fill value to use with fill_mode='constant'
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """

    def __init__(
            self,
            fill_mode='constant',
            interpolation='nearest',
            cval=0,
            relative_translation=True,
    ):
        self.fill_mode = fill_mode
        self.cval = cval
        self.interpolation = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    Args:
        image:        an image
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize=(image.shape[1], image.shape[0]),
        flags=params.cvInterpolation(),
        borderMode=params.cvBorderMode(),
        borderValue=params.cval,
    )

    return output
