import os
import glob

import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import cv2

from pre_processing import apply_transform, adjust_transform_for_image, normalize_image, resize_crop_padding, \
    val_resize_crop_padding, TransformParameters
import params
import random


def fill(img, h, w):
    print(img)
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img


def vertical_shift(img, mask, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img, mask
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h * ratio
    if ratio > 0:
        img = img[:int(h - to_shift), :, :]
        mask = mask[:int(h - to_shift), :]
    if ratio < 0:
        img = img[int(-1 * to_shift):, :, :]
        mask = mask[int(-1 * to_shift):, :]
    img = fill(img, h, w)
    mask = fill(mask, h, w)
    return img, mask


def horizontal_shift(img, mask, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img, mask
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w * ratio
    if ratio > 0:
        img = img[:, :int(w - to_shift), :]
        mask = mask[:, :int(w - to_shift)]
    if ratio < 0:
        img = img[:, int(-1 * to_shift):, :]
        mask = mask[:, int(-1 * to_shift):]
    img = fill(img, h, w)
    mask = fill(mask, h, w)
    return img, mask


def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def add_noise(img):
    VARIABILITY = 50
    deviation = VARIABILITY * random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img


class SEMDataset(Dataset):
    def __init__(
            self,
            image_dir,
            label_dir,
            temp_dir="tmp",
            num_class=3,
            transform_generator=None,
            transform_parameters=None
    ):
        """
        Args:
            image_dir (str): the path where the image is located
            label_dir (str): the path where the mask is located
            transform_generator (): transform the input image
        """
        self.temp_dir = temp_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.num_class = num_class
        self.transform = transforms.ToTensor()

        self.transform_generator = transform_generator
        self.transform_parameters = transform_parameters or TransformParameters()

        self.image_paths = self.check_exist_data()
        self.image_paths.sort()

        self.label_paths = [os.path.join(label_dir, "%s.png" % os.path.splitext(os.path.basename(filepath))[0])
                            for filepath in self.image_paths]

    def get_basename(self, index):
        basename = os.path.basename(self.image_paths[index]).split(".")[0]
        return basename

    def check_exist_data(self):
        all_imgs = []
        img_files = glob.glob(os.path.join(self.image_dir, "*"))
        mask_files = os.listdir(self.label_dir)
        for img_file in img_files:
            img_name_without_ext = os.path.splitext(
                os.path.basename(img_file))[0]
            mask_fname = "{}.png".format(img_name_without_ext)
            if mask_fname in mask_files:
                all_imgs.append(img_file)

        return all_imgs

    def _augment(self, image, label):
        # augment image and label
        if self.transform_generator is not None:
            transform = adjust_transform_for_image(next(self.transform_generator), image,
                                                   self.transform_parameters.relative_translation)
            image = apply_transform(
                transform, image, self.transform_parameters)
            label = apply_transform(
                transform, label, self.transform_parameters)

        return image, label

    def _agument_lib(self, image, label):
        # transform with lib
        import albumentations as album
        transform_generator = album.Compose([
            # album.HorizontalFlip(p=0.5),
            # album.ToSepia(always_apply=False, p=0.15),
            album.RandomBrightnessContrast(always_apply=False, p=0.5, brightness_limit=(-0.1, 0.1),
                                           contrast_limit=(-0.1, 0.1)),
            album.RGBShift(always_apply=False, p=0.5, r_shift_limit=(-20, 20),
                           g_shift_limit=(-20, 20), b_shift_limit=(-20, 20)),
        ])

        transformed = transform_generator(image=image)
        transformed_image = transformed['image']
        # transformed_masks = transformed['masks']
        return transformed_image, label

    def __getitem__(self, index):
        # pre-processing image to target dim
        image, mask = resize_crop_padding(
            self.image_paths[index], self.label_paths[index], self.temp_dir)

        # online augmentation

        image, mask = self._agument_lib(image, mask)
        image, mask = self._augment(image, mask)

        # debug
        # print(image.shape, mask.shape)
        # cv2.imwrite('debug/debug_img_' + str(index) + '.png', image)
        # cv2.imwrite('debug/debug_mask_' + str(index) + '.png', mask)

        # normalize image data
        image = normalize_image(image)

        # one-hot encoding mask
        label = np.zeros(mask.shape[:2] + (self.num_class,))

        for i in range(self.num_class + 1):
            if i == 0 or i == 1:
                (label[..., 0])[mask[..., 0] == i * 100] = 1
            else:
                (label[..., 1])[mask[..., 0] == i * 100] = 1

        return self.transform(image), self.transform(label)

    def __len__(self):
        return len(self.image_paths)


class SEMValDataset(Dataset):
    def __init__(self, image_dir):
        """
        Args:
            image_dir (str): the path where the image is located
            label_dir (str): the path where the mask is located
            transform_generator (str): transform the input image
        """
        self.transform = transforms.ToTensor()

        self.image_paths = glob.glob(os.path.join(image_dir, "*.png"))
        # print(self.image_paths, image_dir)
        self.image_paths.sort()

    def get_basename(self, index):
        basename = os.path.basename(self.image_paths[index]).split(".p")[0]
        return basename

    def __getitem__(self, index):
        image = val_resize_crop_padding(self.image_paths[index])

        # normalize image data
        image = normalize_image(image)

        return self.transform(image)

    def __len__(self):
        return len(self.image_paths)


class SEMValDataset_an_img(Dataset):
    def __init__(self, img, img_path=None):
        """
        Args:
            image_dir (str): the path where the image is located
            label_dir (str): the path where the mask is located
            transform_generator (str): transform the input image
        """
        self.transform = transforms.ToTensor()
        self.extend = ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']
        # self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.img = img
        self.img_path = img_path
        self.target_w = 512
        self.target_h = 288
        self.ratio_w = None
        self.ratio_h = None
        # self.image_paths.sort()

    def get_basename(self):
        basename = os.path.basename(self.image_path).split(".")[0]
        return basename

    def get_org_img(self):
        # image = val_resize_crop_padding(self.image_path)

        return self.img

    def __getitem__(self, index):
        image = self.img.copy()
        self.ratio_w = image.shape[1] / self.target_w
        self.ratio_h = image.shape[0] / self.target_h
        image = cv2.resize(image, (self.target_w, self.target_h))
        # path = self.img_path
        # image = val_resize_crop_padding(path)

        # normalize image data
        image = normalize_image(image)

        return self.transform(image)

    def __len__(self):
        return 1
