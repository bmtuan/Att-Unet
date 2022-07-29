import argparse, sys
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from infordio_ocr.unet_seg.dataset import SEMValDataset, SEMValDataset_an_img
from infordio_ocr.unet_seg.modules import *
from infordio_ocr.unet_seg.save_history import *
from infordio_ocr.unet_seg.advance_model import AttU_Net
from infordio_ocr.unet_seg.base_model import UNet
from infordio_ocr.unet_seg.postProcess import to_rect,crop_image_by_bbox, rotate, cropter
import cv2
import os
import infordio_ocr.unet_seg.params
from infordio_ocr.key_dict_for_get_infor import GN_HONDA1, GN_HONDA3
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from infordio_ocr.conf import CONF_CRAFT
session = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()


model_cut = None
device = None
img_height= None # int(CONF_CRAFT['classify_d']['img_height'])
img_width = None # int(CONF_CRAFT['classify_d']['img_width'])


def save_img(box, img_draw, label, save_dir, basename):

    warp, M = crop_image_by_bbox(img_draw, box)
    if save_dir:
        # color = (255, 0, 255)
        # cv2.drawContours(img_draw, [box.astype(np.int)], -1, color, 3)
        cv2.imwrite(os.path.join(save_dir, "%s_box.png" % basename), warp)
        # label.save(os.path.join(save_dir, "%s.png" % basename))
    return warp


def an_img(img_org,img_path=None, num_workers=0):
    global model_cut
    model = model_cut
    global device
    img= img_org.copy()
    dataset = SEMValDataset_an_img(img, img_path=img_path)
    loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=num_workers, batch_size=1, shuffle=False)
    pbar = tqdm(loader)

    for batch_idx, images in enumerate(pbar):
        images = images.to(device)
        probs = model.forward(images).data.cpu().numpy()  # 1 * C * H * W
        preds = np.argmax(probs, axis=1).astype(np.uint8) * 255  # 1 * H * W
        pred = preds[0, ...]  # H x W
        label = Image.fromarray(pred).convert("L")

        img_draw = np.array(label)
        box = to_rect(img_draw)

        print('ratio_w:',dataset.ratio_w, dataset.ratio_h)
        ret_box = []
        for p in box:
            ret_box.append([p[0]*dataset.ratio_w, p[1]*dataset.ratio_h])
        ret_box = np.array(ret_box)

        startidx = ret_box.sum(axis=1).argmin()
        ret_box = np.roll(ret_box, 4-startidx, 0)
        # print(ret_box)

        return ret_box, label

def an_img2(img_org,img_path=None, num_workers=0):
    global model_cut
    model = model_cut
    global device
    img= img_org.copy()
    dataset = SEMValDataset_an_img(img, img_path=img_path)
    loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=num_workers, batch_size=1, shuffle=False)
    pbar = tqdm(loader)

    for batch_idx, images in enumerate(pbar):
        images = images.to(device)
        probs = model.forward(images).data.cpu().numpy()  # 1 * C * H * W
        preds = np.argmax(probs, axis=1).astype(np.uint8) * 255  # 1 * H * W
        pred = preds[0, ...]  # H x W
        label = Image.fromarray(pred).convert("L")

        img_draw = np.array(label)
        img_rotated = rotate(img_draw, img)

        return img_rotated

def pading(img, target_w, target_h, padall=False):
    img_h, img_w = img.shape[:2]
    print('abs(img_w/img_h - target_w/target_h):', abs(img_w / img_h - target_w / target_h))
    padall = True
    if abs(img_w/img_h - target_w/target_h) >= 0.5 or padall:
        ratio = img_h/target_h if img_h/target_h > img_w/target_w else img_w/target_w
        pad_img = np.full((int(ratio*target_h), int(ratio*target_w), 3), 200, dtype=np.uint8)
        sx = (pad_img.shape[1] - img_w) //2
        sy = (pad_img.shape[0] - img_h) //2
        pad_img[sy:sy+img_h, sx:sx+img_w] = img
        return pad_img
    else:
        return img

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def an_img3(img_org,img_path=None, num_workers=0, brand=None, out_dir_debug=None, fname=None):
    global model_cut
    model = model_cut
    global device
    img= img_org.copy()
    if brand  in [ GN_HONDA1]: # GN_HONDA3
        img= pading(img, 640, 480)
        if out_dir_debug:
            cv2.imwrite(os.path.join(out_dir_debug, '{}_pading.jpg'.format(fname)), img)
    dataset = SEMValDataset_an_img(img, img_path=img_path)
    loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=num_workers, batch_size=1, shuffle=False)
    pbar = tqdm(loader)
    print(pbar)
    for batch_idx, images in enumerate(pbar):
        images = images.to(device)
        probs = model.forward(images).data.cpu().numpy()  # 1 * C * H * W
        preds = np.argmax(probs, axis=1).astype(np.uint8) * 255  # 1 * H * W
        pred = preds[0, ...]  # H x W
        label = Image.fromarray(pred).convert("L")

        img_draw = np.array(label)
        print('ratio_w:',dataset.ratio_w, dataset.ratio_h)
        if out_dir_debug:
            cv2.imwrite(os.path.join(out_dir_debug, '{}_cut.jpg'.format(fname)), img_draw)

        # kernel = np.ones((5, 5), np.uint8)
        # img_draw = cv2.dilate(img_draw, kernel, iterations=1)

        # test crop quadrangtle box
        contours, hierarchy = cv2.findContours(img_draw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print('len contours:  ', len(contours))

        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        max_area = areas[max_index]

        img_draw_copy = np.stack((img_draw,) * 3, axis=-1)
        hull = cv2.convexHull(cnt)
        cv2.drawContours(img_draw_copy, [hull], 0, (255, 0, 0), 2)
        cnt = hull

        epsilon = 0.08 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        print(approx, approx[0])
        print(img_draw_copy.shape)
        print(img.shape)
        print('hulll: ', hull)

        cv2.drawContours(img_draw_copy, [cnt], 0, (0, 255, 0), 3)
        cv2.drawContours(img_draw_copy, [approx], 0, (0, 0, 255), 3)
        app_area = cv2.contourArea(approx)

        cv2.imwrite(os.path.join('/home/aimenext2/output/test_warp', fname + '_hitmap' + '.jpg'), img_draw_copy)
        print("simplified contour has", len(approx), "points")
        import random
        if len(approx) == 4:
            for point in approx:
                point[0][0] *= dataset.ratio_w
                point[0][1] *= dataset.ratio_h
            four_point =np.asarray([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
            warped = four_point_transform(img, four_point)
            cv2.imwrite(os.path.join('/home/aimenext2/output/test_warp', fname + '.jpg'), warped)
        else:
            warped = img_draw_copy
            cv2.imwrite(os.path.join('/home/aimenext2/output/test_warp', fname + '.jpg'), warped)
        # cv2.imshow('image', warped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        if app_area/max_area > 0.9:
            return  warped
        ret = cropter(img_draw, img, dataset.ratio_w, dataset.ratio_h, brand=brand)

        return ret
def load_model(model='AttU_Net', snapshot='/mnt/disk1/hunglv/model_checkpoints/hoyu_vin/bce/AttU_Net_99.pth', num_class=2):

    # device = "cpu"        print('ratio_w:',dataset.ratio_w, dataset.ratio_h)

    device = torch.device("cuda:%d" % 0 if torch.cuda.is_available() else "cpu")
    print(device)
    # init model
    if model == "AttU_Net":
        model = AttU_Net(in_channels=3, n_classes=num_class)
    else:
        # default is classical UNet
        model = UNet(in_channels=3, n_classes=num_class, depth=2, batch_norm=True, padding=True)
    model = model.to(device)
    load_checkpoint(snapshot, model, None, device=device)
    model.eval()
    return model, device

inited_model = False
def init_model():
    global inited_model
    assert inited_model is False, "cut broad Model loaded. Dont call init_model again!"
    inited_model = True

    global img_height, img_width

    # # classify_checkpoint = '/home/ubuntu/nhinhlt/classify/checkpoint/direction0203_64_480_640_adam_20200302103309/weights.014-0.40276.hdf5'  #85 73 85 85
    # img_height = 480
    # img_width = 640
    #
    classify_checkpoint = str(CONF_CRAFT['cut_broad']['checkpoint'])
    img_height = int(CONF_CRAFT['cut_broad']['img_height'])
    img_width = int(CONF_CRAFT['cut_broad']['img_width'])

    global model_cut, device
    global session
    if classify_checkpoint != 'False' and img_height is not None and img_width is not None:
        print('cut broad checkpoint', classify_checkpoint)
        set_session(session)
        model_cut, device = load_model(snapshot=classify_checkpoint)
    else:
        model_cut, device = None, None

#     predict2(args)
    # python predict.py --model AttU_Net --snapshot /mnt/disk1/hunglv/model_checkpoints/hoyu_vin/bce/AttU_Net_99.pth --gpu_id 1 --input_dir /mnt/disk1/nhinhlt/datasets/input/test  --save_dir /mnt/disk1/nhinhlt/datasets/output/test_unet-seg

def adir(in_dir, out):
    if not os.path.exists(out):
        os.mkdir(out)
    imgs = os.listdir(in_dir)
    for idx, fn in enumerate(imgs):
        print('{}/{}:{}'.format(idx, len(imgs), fn))
        src_path = os.path.join(in_dir, fn)
        basename = os.path.splitext(fn)[0]
        dst_path = os.path.join(out, '{}.jpg'.format(basename))
        if os.path.exists(dst_path):
            continue

        img = cv2.imread(src_path)
        if img is None:
            print('img is ', img)
            continue
        warp = an_img3(img)

        cv2.imwrite(dst_path, warp)

def main(datap ='', out = ''):
    dataname = os.path.basename(datap) + '_crop'
    out = os.path.join(out, dataname)
    print(dataname)
    print(out)
    if not os.path.exists(out):
        os.mkdir(out)
    sub_folders = ['train', 'validation']

    for subfoldern in sub_folders:
        print('subfoldern:', subfoldern)
        subfolderp = os.path.join(datap, subfoldern)
        subfolder_out = os.path.join(out, subfoldern)
        if not os.path.exists(subfolder_out):
            os.mkdir(subfolder_out)
        brands = [d for d in os.listdir(subfolderp) if os.path.isdir(os.path.join(subfolderp,d))]

        print(brands)
        for brandn in brands:
            print('brandn:', brandn)
            brandp = os.path.join(subfolderp, brandn)
            brand_out = os.path.join(subfolder_out, brandn)
            adir(brandp, brand_out)

    print(dataname)
    print(out)

if __name__ == "__main__":
    init_model()
    P1 = '/home/ubuntu/nhinhlt/orientation_detect/predict/datasets/classify_boards_070420_not_remove'
    P2 = '/home/ubuntu/nhinhlt/orientation_detect/predict/datasets/classify_1407'
    out = '/home/ubuntu/nhinhlt/orientation_detect/predict/dataset2'
    main(datap=P1, out=out)
    main(datap=P2, out=out)

    # CUDA_VISIBLE_DEVICES=0 python -m infordio_ocr.unet_seg.init_cut_broad_model