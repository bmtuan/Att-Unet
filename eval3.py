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
import cv2

import math

def get_intersect_int(a1, a2, b1, b2):
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
    return np.array([[int(x/z), int(y/z)]])

def approx_process(approx):
    approx = list(approx)
    list_point = approx + [approx[0]]
    list_line = []
    for i in range(len(list_point) - 1):
        list_line.append([list_point[i], list_point[i + 1]])
    while len(list_line)>4:
        list_length = [distance(a[0][0], a[1][0]) for a in list_line]
        list_length = np.asarray(list_length)
        min_idx = np.argmin(list_length, axis=0)

        # line_deleted = list_line[min_idx]

        if min_idx == 0:
            before_idx = len(list_line)
            after_idx = 1
        elif min_idx == len(list_line) - 1:
            before_idx = len(list_line) - 1
            after_idx = 0
        else:
            before_idx = min_idx - 1
            after_idx = min_idx + 1

        line_before = list_line[before_idx]
        line_after = list_line[after_idx]
        point_intersect = get_intersect_int(line_before[0][0], line_before[1][0], line_after[0][0], line_after[1][0])

        # change point of line
        new_line_before = [line_before[0], point_intersect]
        new_line_after = [point_intersect, line_after[1]]

        list_line[before_idx] = new_line_before
        list_line[after_idx] = new_line_after
        list_line.pop(min_idx)

    list_point_final = []
    for line in list_line:
        list_point_final.append(line[0])
        list_point_final.append(line[1])
    point_new = np.unique(list_point_final, axis=0)
    return point_new

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

path = '/media/aimenext2/Newdisk/chien_KAIHO/dataset/data_classify_board/catergory'
new_path = '/media/aimenext2/Newdisk/chien_KAIHO/dataset/data_classify_board/catergory_cuts'


def eval(args):
    # init model
    # model = UNet(in_channels=3, n_classes=args.num_class, depth=2, batch_norm=True, padding=True)

    model = AttU_Net(in_channels=3, n_classes=args.num_class)
    model = model.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_checkpoint(args.snapshot, model, None)
    SAVE_PATH = '/home/aimenext2/output/test_warp3'

    # path_src = '/media/aimenext2/Newdisk/chien_KAIHO/dataset/data_classify_board/catergory'
    path_src = '/media/aimenext2/Newdisk/chien_KAIHO/dataset/data_classify_board/catergory_cuts2_splited'
    # new_path = '/media/aimenext2/Newdisk/chien_KAIHO/dataset/data_classify_board/catergory_cuts2'
    new_path = '/media/aimenext2/Newdisk/chien_KAIHO/dataset/data_classify_board/catergory_cut_split'

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    num_warp = 0
    num_parallel = 0

    for folder in os.listdir(path_src):
        folder_src = os.path.join(path_src, folder)
        folder_dst = os.path.join(new_path, folder)
        if not os.path.exists(folder_dst):
            os.mkdir(folder_dst)


        # for path in os.listdir(args.input_dir):
        for path in os.listdir(folder_src):
            print(path)
            img_path = os.path.join(folder_src, path)
            img = cv2.imread(img_path)
            cv2.imwrite(os.path.join(folder_dst, path.split('.')[0] + '_input.jpg'), img)
            height, width = img.shape[:2]
            dim_origin = (width, height)

            dataset = SEMValDataset_an_img(img, img_path=img_path)
            loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=1, shuffle=False)
            pbar = tqdm(loader)
            # print(pbar)
            for batch_idx, images in enumerate(pbar):
                images = images.to(device)
                probs = model.forward(images).data.cpu().numpy()  # 1 * C * H * W
                preds = np.argmax(probs, axis=1).astype(np.uint8) * 255  # 1 * H * W
                pred = preds[0, ...]  # H x W
                label = Image.fromarray(pred).convert("L")
                mask = np.array(label)
                cv2.imwrite(os.path.join(folder_dst, path.split('.')[0] + '_mask.jpg'), mask)

                ###
                mask = cv2.resize(mask, dim_origin, interpolation=cv2.INTER_AREA)

                size_open = 150
                open_mask = cv2.copyMakeBorder(mask, size_open, size_open, size_open, size_open, cv2.BORDER_CONSTANT,
                                               value=(0, 0, 0))
                open_origin_img = cv2.copyMakeBorder(img, size_open, size_open, size_open, size_open, cv2.BORDER_CONSTANT,
                                                     value=(0, 0, 0))

                blank_image_for_draw = np.zeros((height + size_open * 2, width + size_open * 2), np.uint8)

                contours, hierarchy = cv2.findContours(open_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # print('len contours:  ', len(contours))

                # try:
                areas = [cv2.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                cnt = contours[max_index]

                img_draw_copy = np.stack((blank_image_for_draw,) * 3, axis=-1)
                img_draw_copy2 = np.stack((blank_image_for_draw,) * 3, axis=-1)
                img_draw_copy3 = blank_image_for_draw

                cv2.drawContours(img_draw_copy3, [cnt], contourIdx=-1, color=(255, 255, 255), thickness=-1)
                mask = img_draw_copy3
                # cv2.imwrite(os.path.join(out_dir_debug, '{}_mask_old.jpg'.format(fname)), mask)
                kernel = np.ones((150, 150), np.uint8)
                dilation = cv2.dilate(mask, kernel, iterations=1)
                # cv2.imwrite(os.path.join(out_dir_debug, '{}_mask_old2.jpg'.format(fname)), dilation)

                kernel = np.ones((3, 3), np.uint8)
                mask = dilation
                for i in range(45):
                    mask = cv2.erode(dilation, kernel, iterations=1)

                # mask = cv2.erode(dilation, kernel, iterations=1)



                cv2.imwrite(os.path.join(SAVE_PATH, path), mask)

                # print('shape mask:', mask.shape)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # print('len contours2:  ', len(contours))

                # areas = [cv2.contourArea(c) for c in contours]
                # max_index = np.argmax(areas)
                # cnt = contours[max_index]
                # max_area = areas[max_index]
                cnt = contours[0]
                max_area = cv2.contourArea(cnt)
                # print('donee')

                hull = cv2.convexHull(cnt)

                # a branch for find min are parallelogram
                list_point_hull = [a[0] for a in hull]
                area, v1, v2, v3, v4, x1, x2 = mep(list_point_hull)
                list_point = [v1, v2, v3, v4]
                for a in list_point:
                    x1 = int(a[0])
                    x2 = int(a[1])
                    cv2.circle(img_draw_copy2, (x1, x2), 10, (255, 255, 255), 10)

                four_point = np.asarray([list_point[0], list_point[1], list_point[2], list_point[3]])
                warped_parallel = four_point_transform(open_origin_img, four_point)

                cnt = hull
                epsilon = 0.03 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                approx = approx_process(approx)
                ## draw approximate polygon
                # cv2.drawContours(img_draw_copy, [approx], 0, (0, 0, 255), 3)
                app_area = cv2.contourArea(approx)

                if len(approx) == 4:
                    four_point = np.asarray([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
                    warped = four_point_transform(open_origin_img, four_point)
                    # cv2.imwrite(os.path.join(out_dir_debug, fname + '.jpg'), warped)
                else:
                    print('fail')
                    warped = warped_parallel
                    # cv2.imwrite(os.path.join(out_dir_debug, fname + '.jpg'), warped)

                if app_area / max_area > 0.93:
                    # return warped
                    ret =warped
                    num_warp += 1
                else:
                # ret = cropter_origin(open_mask, open_origin_img)
                    ret = warped_parallel
                    num_parallel +=1
                # except:
                #     # ret = cropter_origin(open_mask, open_origin_img)
                #     print('failll')
                #     ret = cropter_origin(open_mask, open_origin_img)
                print(os.path.join(folder_dst, path.split('.')[0] + '.jpg'))
                cv2.imwrite(os.path.join(folder_dst, path.split('.')[0] + '_output.jpg'), ret)
    print('ti le: {}'.format(num_warp/(num_warp+num_parallel)))

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--input_dir', type=str)
    parser.add_argument('-s', '--snapshot', type=str)
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-nc', '--num_class', type=int, default=2)
    args = parser.parse_args()

    eval(args)