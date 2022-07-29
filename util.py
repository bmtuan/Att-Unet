import cv2
import os
import numpy as np
import time
from tqdm import tqdm


def get_image_data(masks_path, images_path, output_path):
    list_masks = os.listdir(masks_path)
    for index, mask in enumerate(list_masks):
        print(images_path + mask[:-4] + '.jpg')
        img = cv2.imread(masks_path + mask[:-4] + '.jpg')
        cv2.imwrite(output_path + mask[:-4] + '.png', img)


def extract_frame(video_path, video_name, output_path, frequency):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cap = cv2.VideoCapture(os.path.join(video_path,video_name))
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % frequency == 0:
            cv2.imwrite(output_path + video_name[:-4] + '_' + str(i) + '.png', frame)
        # print(output_path + video_name[:-4] + '_' + str(i) + '.png')
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def train_test_split(image_folder_path, mask_folder_path, train_folder_path, val_folder_path):
    list_mask_paths = os.listdir(mask_folder_path)
    for index, image_path in tqdm(enumerate(list_mask_paths)):
        image = cv2.imread(image_folder_path + image_path)
        mask = cv2.imread(mask_folder_path + image_path)
        if index % 5 == 1:
            cv2.imwrite(val_folder_path + 'image/' + image_path, image)
            cv2.imwrite(val_folder_path + 'mask/' + image_path, mask)
        else:
            cv2.imwrite(train_folder_path + 'image/' + image_path, image)
            cv2.imwrite(train_folder_path + 'mask/' + image_path, mask)


def create_blank_image(image, image_path, mask_path):
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    # image = np.zeros((512, 288, 3), np.uint8)
    mask = np.zeros((1080, 1920, 1), np.uint8)
    # for i in range(20):
    # cv2.imwrite(f'{image_path}{i}.png', image)
    cv2.imwrite(f'{mask_path}{image}', mask)


if __name__ == '__main__':
    train_test_split(
        image_folder_path='/media/aimenext/disk1/tuanbm/pupil_segmentation/iris_pupil_annotation_tool/data_0707/',
        mask_folder_path='/media/aimenext/disk1/tuanbm/pupil_segmentation/iris_pupil_annotation_tool/mask_0707/',
        train_folder_path='/media/aimenext/disk1/tuanbm/pupil_segmentation/unet/data/train/',
        val_folder_path='/media/aimenext/disk1/tuanbm/pupil_segmentation/unet/data/val/'
    )

    # get_image_data(
    #     masks_path='/home/bmtuan/Desktop/data_aims/output/',
    #     images_path='/home/bmtuan/Desktop/test/',
    #     output_path='/home/bmtuan/Desktop/test/'
    # )
    # list_video_names = os.listdir('/media/aimenext/disk1/tuanbm/pupil_segmentation/data_aims_video/0707')
    # # print(list_video_names)
    # for video_name in tqdm(list_video_names):
    #     print(video_name)
    #     extract_frame(
    #         video_path='/media/aimenext/disk1/tuanbm/pupil_segmentation/data_aims_video/0707/',
    #         video_name=video_name,
    #         output_path=f'/media/aimenext/disk1/tuanbm/pupil_segmentation/iris_pupil_annotation_tool/full_data_0707/{video_name[:-4]}/',
    #         frequency=1
    #     )
    # path = '../iris_pupil_annotation_tool/data/'
    # for image in os.listdir(path):
    #     create_blank_image(
    #         image=image,
    #         image_path='/home/bmtuan/Desktop/image/',
    #         mask_path='./data/train/mask/'
    #     )
