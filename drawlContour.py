import cv2
import os
from tqdm import tqdm


def drawl(origin_image, mask, mask_path, output_path):
    img_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    origin_image = cv2.resize(src=origin_image, dsize=(512, 288))
    ret, thresh_pupil = cv2.threshold(img_gray, 150, 250, cv2.THRESH_BINARY)
    ret_, thresh_iris = cv2.threshold(img_gray, 50, 140, cv2.THRESH_BINARY)

    contours_pupil, hierarchy = cv2.findContours(thresh_pupil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_iris, hierarchy = cv2.findContours(thresh_iris, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(origin_image, contours_pupil, -1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.drawContours(origin_image, contours_iris, -1, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(output_path + mask_path, origin_image)


if __name__ == '__main__':
    # for folder in os.listdir('./data/test/0811/predict/'):
    #     list_mask = os.listdir('./data/test/0811/predict/' + folder)
    #
    #     for mask_path in tqdm(list_mask):
    #         mask = cv2.imread(f'./data/test/0811/predict/{folder}/{mask_path}')
    #         origin_image = cv2.imread(f'./data/test/0811/image/{folder}/{mask_path}')
    #         output_dir = f'./data/test/0811/drawl/{folder}/'
    #
    #         if not os.path.exists(output_dir):
    #             os.makedirs(output_dir)
    #         drawl(
    #             origin_image=origin_image,
    #             mask=mask,
    #             mask_path=mask_path,
    #             output_path=output_dir
    #         )

    list_mask = os.listdir('./data/test/predict_/')
    # print(list_mask[1])
    for mask_path in tqdm(list_mask):
        mask = cv2.imread(f'./data/test/predict_/{mask_path}')
        origin_image = cv2.imread(f'./data/test/image_/{mask_path}')
        output_dir = f'./data/test/drawl_/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        drawl(
            origin_image=origin_image,
            mask=mask,
            mask_path=mask_path,
            output_path=output_dir
        )
