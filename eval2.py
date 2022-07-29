import argparse, sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from dataset import SEMDataset, SEMValDataset, SEMValDataset_an_img
from modules import *
from save_history import *
from advance_model import AttU_Net
import params
from base_model import UNet
import cv2

import math

def distance(p1, p2, p):
    return abs(((p2[1]-p1[1])*p[0] - (p2[0]-p1[0])*p[1] + p2[0]*p1[1] - p2[1]*p1[0]) /
        math.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2))

def antipodal_pairs(convex_polygon):
    l = []
    n = len(convex_polygon)
    p1, p2 = convex_polygon[0], convex_polygon[1]

    t, d_max = None, 0
    for p in range(1, n):
        d = distance(p1, p2, convex_polygon[p])
        if d > d_max:
            t, d_max = p, d
    l.append(t)

    for p in range(1, n):
        p1, p2 = convex_polygon[p % n], convex_polygon[(p+1) % n]
        _p, _pp = convex_polygon[t % n], convex_polygon[(t+1) % n]
        while distance(p1, p2, _pp) > distance(p1, p2, _p):
            t = (t + 1) % n
            _p, _pp = convex_polygon[t % n], convex_polygon[(t+1) % n]
        l.append(t)

    return l

def mep(convex_polygon):
    def compute_parallelogram(convex_polygon, l, z1, z2):
        def parallel_vector(a, b, c):
            v0 = [c[0]-a[0], c[1]-a[1]]
            v1 = [b[0]-c[0], b[1]-c[1]]
            return [c[0]-v0[0]-v1[0], c[1]-v0[1]-v1[1]]

        # finds intersection between lines, given 2 points on each line.
        # (x1, y1), (x2, y2) on 1st line, (x3, y3), (x4, y4) on 2nd line.
        def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
            px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4))/((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
            py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4))/((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
            return px, py


        # from each antipodal point, draw a parallel vector,
        # so ap1->ap2 is parallel to p1->p2
        #    aq1->aq2 is parallel to q1->q2
        p1, p2 = convex_polygon[z1 % n], convex_polygon[(z1+1) % n]
        q1, q2 = convex_polygon[z2 % n], convex_polygon[(z2+1) % n]
        ap1, aq1 = convex_polygon[l[z1 % n]], convex_polygon[l[z2 % n]]
        ap2, aq2 = parallel_vector(p1, p2, ap1), parallel_vector(q1, q2, aq1)

        a = line_intersection(p1[0], p1[1], p2[0], p2[1], q1[0], q1[1], q2[0], q2[1])
        b = line_intersection(p1[0], p1[1], p2[0], p2[1], aq1[0], aq1[1], aq2[0], aq2[1])
        d = line_intersection(ap1[0], ap1[1], ap2[0], ap2[1], q1[0], q1[1], q2[0], q2[1])
        c = line_intersection(ap1[0], ap1[1], ap2[0], ap2[1], aq1[0], aq1[1], aq2[0], aq2[1])

        s = distance(a, b, c) * math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
        return s, a, b, c, d


    z1, z2 = 0, 0
    n = len(convex_polygon)

    # for each edge, find antipodal vertice for it (step 1 in paper).
    l = antipodal_pairs(convex_polygon)

    so, ao, bo, co, do, z1o, z2o = 100000000000, None, None, None, None, None, None

    # step 2 in paper.
    for z1 in range(0, n):
        if z1 >= z2:
            z2 = z1 + 1
        p1, p2 = convex_polygon[z1 % n], convex_polygon[(z1+1) % n]
        a, b, c = convex_polygon[z2 % n], convex_polygon[(z2+1) % n], convex_polygon[l[z2 % n]]
        if distance(p1, p2, a) >= distance(p1, p2, b):
            continue

        while distance(p1, p2, c) > distance(p1, p2, b):
            z2 += 1
            a, b, c = convex_polygon[z2 % n], convex_polygon[(z2+1) % n], convex_polygon[l[z2 % n]]

        st, at, bt, ct, dt = compute_parallelogram(convex_polygon, l, z1, z2)

        if st < so:
            so, ao, bo, co, do, z1o, z2o = st, at, bt, ct, dt, z1, z2

    return so, ao, bo, co, do, z1o, z2o

def four_point_transform(image, pts):
    rect = order_points(pts)
    if list(rect[0]) == list(rect[1]) or list(rect[0]) == list(rect[2]) or list(rect[0]) == list(rect[3]):
        rect2 = np.zeros((4, 2), dtype="float32")
        rect2[0] = pts[3]
        rect2[1] = pts[2]
        rect2[2] = pts[1]
        rect2[3] = pts[0]
        rect = rect2
    # try:
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
    pts = np.asarray(pts)
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def cropter_origin(pred, org_img):
    return org_img


def get_intersect_int2(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        # return (float('inf'), float('inf'))
        return False
    return np.array([int(x/z), int(y/z)])


def distance_app(point1, point2):
    return int(math.sqrt((point2[1] - point1[1]) ** 2 + (point2[0] - point1[0]) ** 2))


def get_point(approx):
    return [list(a[0]) for a in approx]


def get_line(points):
    points = points + [points[0]]
    list_line = []
    for i in range(len(points) - 1):
        list_line.append([points[i], points[i + 1]])
    return list_line


def get_length(lines):
    return [distance_app(line[0], line[1]) for line in lines]


def get_intersect(lines, minx, maxx, miny, maxy, thr=150):
    list_intersect = []
    for idx in range(len(lines)):
        line_deleted = lines[idx]

        if idx == 0:
            before_idx = len(lines) - 1
            after_idx = 1
        elif idx == len(lines) - 1:
            before_idx = len(lines) - 2
            after_idx = 0
        else:
            before_idx = idx - 1
            after_idx = idx + 1

        line_before = lines[before_idx]
        line_after = lines[after_idx]

        point_intersect = get_intersect_int2(line_before[0], line_before[1], line_after[0], line_after[1])

        if point_intersect[0] > minx - thr and point_intersect[0] < maxx + thr and point_intersect[1] > miny - thr and \
                point_intersect[1] < maxy + thr:
            list_intersect.append(point_intersect)
        else:
            list_intersect.append(None)
    return list_intersect


def length_to_center(intersects, center):
    distance_list = []
    for i in range(len(intersects)):
        if intersects[i] is not None:
            distance_list.append(distance_app(intersects[i], center))
        else:
            distance_list.append(99999999)

    return distance_list


def approx_process2(approx):
    cur_approx = approx
    while len(cur_approx) > 4:
        M = cv2.moments(cur_approx)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center = [cX, cY]

        points = get_point(cur_approx)

        list_x = [a[0] for a in points]
        list_y = [a[1] for a in points]

        minx = min(list_x)
        maxx = max(list_x)
        miny = min(list_y)
        maxy = max(list_y)

        lines = get_line(points)
        length = get_length(lines)
        intersects = get_intersect(lines, minx, maxx, miny, maxy)
        leng_center = length_to_center(intersects, center)

        total_length = []
        for i in range(len(length)):
            total_length.append(length[i] + leng_center[i])

        idx_point_chosen = np.argmin(total_length)
        if total_length[idx_point_chosen] < 9999999:
            deleted_line = lines[idx_point_chosen]

            cur_approx = list(cur_approx)
            if idx_point_chosen != len(lines) - 1:
                del cur_approx[idx_point_chosen + 1]
                del cur_approx[idx_point_chosen]
            else:
                del cur_approx[idx_point_chosen]
                del cur_approx[0]

            cur_approx += [[np.array(intersects[idx_point_chosen], dtype=np.int32)]]
            cur_approx = np.array(cur_approx)
            cur_approx = cv2.convexHull(cur_approx)
        else:
            print('cant add')
            break
    return cur_approx


import time

def eval(args):
    # init model
    # model = UNet(in_channels=3, n_classes=args.num_class, depth=5, batch_norm=False, padding=False)

    model = AttU_Net(in_channels=3, n_classes=args.num_class)
    model = model.cuda()
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    load_checkpoint(args.snapshot, model, None)
    SAVE_PATH = '/media/aimenext2/Newdisk/chien_KAIHO/unet/unet-seg/output'

    print(os.listdir(args.input_dir))
    for path in os.listdir(args.input_dir):
        print(path)
        start = time.time()
        img_path = os.path.join(args.input_dir, path)
        img = cv2.imread(img_path)

        height, width = img.shape[:2]
        dim_origin = (width, height)

        dataset = SEMValDataset_an_img(img, img_path=img_path)
        loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=1, shuffle=False)
        pbar = tqdm(loader)
        # print(pbar)
        for batch_idx, images in enumerate(pbar):
            images = images.to(device)
            with torch.no_grad():
                probs = model.forward(images).data.cpu().numpy()  # 1 * C * H * W
            preds = np.argmax(probs, axis=1).astype(np.uint8) * 255  # 1 * H * W
            pred = preds[0, ...]  # H x W
            label = Image.fromarray(pred).convert("L")
            mask = np.array(label)

            cv2.imwrite(os.path.join(SAVE_PATH, path), mask)
            # continue
        print(time.time()-start)
            # mask = cv2.resize(mask, dim_origin, interpolation=cv2.INTER_AREA)
            #
            # # cv2.imwrite(os.path.join(out_dir_debug, '{}_mask.jpg'.format(fname)), mask)
            # size_open = 150
            # open_mask = cv2.copyMakeBorder(mask, size_open, size_open, size_open, size_open, cv2.BORDER_CONSTANT,
            #                                value=(0, 0, 0))
            # open_origin_img = cv2.copyMakeBorder(img, size_open, size_open, size_open, size_open, cv2.BORDER_CONSTANT,
            #                                      value=(0, 0, 0))
            # blank_image_for_draw = np.zeros((height + size_open * 2, width + size_open * 2), np.uint8)
            # contours, hierarchy = cv2.findContours(open_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # try:
            #     areas = [cv2.contourArea(c) for c in contours]
            #     max_index = np.argmax(areas)
            #     cnt = contours[max_index]
            #
            #     img_draw_copy2 = np.stack((blank_image_for_draw,) * 3, axis=-1)
            #     img_draw_copy3 = blank_image_for_draw
            #     cv2.drawContours(img_draw_copy3, [cnt], contourIdx=-1, color=(255, 255, 255), thickness=-1)
            #
            #     mask = img_draw_copy3
            #
            #     kernel = np.ones((33, 33), np.uint8)
            #     dilation = cv2.dilate(mask, kernel, iterations=1)
            #     kernel = np.ones((3, 3), np.uint8)
            #     mask = dilation
            #     for i in range(11):
            #         mask = cv2.erode(dilation, kernel, iterations=1)
            #
            #     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #
            #     cnt = contours[0]
            #     max_area = cv2.contourArea(cnt)
            #
            #     hull = cv2.convexHull(cnt)
            #
            #     cnt = hull
            #     epsilon = 0.01 * cv2.arcLength(cnt, True)
            #     approx = cv2.approxPolyDP(cnt, epsilon, True)
            #
            #     height_open, width_open = open_origin_img.shape[:2]
            #     approx = approx_process2(approx)
            #
            #     hull = approx
            #     # a branch for find min are parallelogram
            #     # list_point_hull = [a[0] for a in hull]
            #     # area, v1, v2, v3, v4, x1, x2 = mep(list_point_hull)
            #     # list_point = [v1, v2, v3, v4]
            #     # for a in list_point:
            #     #     x1 = int(a[0])
            #     #     x2 = int(a[1])
            #     #     cv2.circle(img_draw_copy2, (x1, x2), 10, (255, 255, 255), 10)
            #     #
            #     # four_point = np.asarray([list_point[0], list_point[1], list_point[2], list_point[3]])
            #     # warped_parallel = four_point_transform(open_origin_img, four_point)
            #
            #
            #     app_area = cv2.contourArea(approx)
            #
            #     if len(approx) == 4:
            #         four_point = np.asarray([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
            #         warped = four_point_transform(open_origin_img, four_point)
            #     else:
            #         warped = warped_parallel
            #
            #     print(app_area / max_area, len(approx))
            #     if app_area / max_area > 0.7:
            #         ret =  warped
            #     else:
            #         ret = warped_parallel
            # except:
            #     print('fail')
            #     ret = cropter_origin(open_mask, open_origin_img)
            #
            # # return ret
            # cv2.imwrite(os.path.join(SAVE_PATH2, path), ret)


            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--input_dir', type=str, default = '/media/aimenext2/Newdisk/chien_KAIHO/unet/unet-seg/data2/Kvasir-SEG/train/imgs')
    parser.add_argument('-s', '--snapshot', type=str, default = 'bce/dbce/AttU_Net_final.pth')
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-nc', '--num_class', type=int, default=2)
    args = parser.parse_args()

    eval(args)