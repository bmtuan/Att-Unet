import argparse, sys
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

from dataset import SEMValDataset
from modules import *
from save_history import *
from advance_model import AttU_Net
from base_model import UNet
from drawlContour import drawl
import params
import time


def predict(args):
    # st = time.time()
    # for input_folder in tqdm(os.listdir(args.folder_dir)):
    input_dir = args.input_dir
    output_dir = args.save_dir
    # input_dir = args.folder_dir + input_folder
    # output_dir = args.save_dir + input_folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = SEMValDataset(os.path.join(input_dir))

    loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=1, shuffle=False)

    device = torch.device("cuda:%d" % args.gpu_id if torch.cuda.is_available() else "cpu")

    # init model
    if args.model == "AttU_Net":
        print('AttU_Net')
        model = AttU_Net(in_channels=3, n_classes=args.num_class)
    else:
        # default is classical UNet
        model = UNet(in_channels=3, n_classes=args.num_class, depth=5, batch_norm=True, padding=True)
    model = model.to(device)

    load_checkpoint(args.snapshot, model, None)

    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader)
        for batch_idx, images in enumerate(pbar):
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.cpu().numpy()
            predicted = np.argmax(outputs, axis=1).astype(np.uint8)
            predicted = predicted[0, ...]
            label = Image.fromarray(predicted * 100).convert("L")
            basename = dataset.get_basename(batch_idx)
            label.save(os.path.join(output_dir, "%s.png" % basename))
    # print(time.time() - st)


def multi_predict(args):
    device = torch.device("cuda:%d" % args.gpu_id if torch.cuda.is_available() else "cpu")
    # init model
    if args.model == "AttU_Net":
        print('AttU_Net')
        model = AttU_Net(in_channels=3, n_classes=args.num_class)
    else:
        # default is classical UNet
        model = UNet(in_channels=3, n_classes=args.num_class, depth=5, batch_norm=True, padding=True)
    model = model.to(device)

    load_checkpoint(args.snapshot, model, None)

    for input_folder in tqdm(os.listdir(args.folder_dir)):
        input_dir = args.folder_dir + input_folder
        output_dir = args.save_dir + input_folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = SEMValDataset(os.path.join(input_dir))

        loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=1, shuffle=False)

        model.eval()
        with torch.no_grad():
            pbar = tqdm(loader)
            for batch_idx, images in enumerate(pbar):
                images = images.to(device)
                outputs = model(images)
                outputs = outputs.cpu().numpy()
                predicted = np.argmax(outputs, axis=1).astype(np.uint8)
                predicted = predicted[0, ...]
                label = Image.fromarray(predicted * 100).convert("L")
                basename = dataset.get_basename(batch_idx)
                label.save(os.path.join(output_dir, "%s.png" % basename))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model', type=str, default="AttU_Net")
    # parser.add_argument('-id', '--input_dir', type=str, default="./data/test/image_/")
    # parser.add_argument('-fd', '--folder_dir', type=str, default="./data/test/image/")
    # parser.add_argument('-sd', '--save_dir', type=str, default="./data/test/predict_/")
    # parser.add_argument('-s', '--snapshot', type=str, default="./bce/2211/AttU_Nettt_97.pth")
    # parser.add_argument('-nw', '--num_workers', type=int, default=0)
    # parser.add_argument('-nc', '--num_class', type=int, default=2)
    # parser.add_argument('-gid', '--gpu_id', type=int, default=0)
    # args = parser.parse_args()
    # predict(args)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="AttU_Net")
    # parser.add_argument('-id', '--input_dir', type=str, default="./data/test/image_/")
    parser.add_argument('-fd', '--folder_dir', type=str, default="./data/test/image/")
    parser.add_argument('-sd', '--save_dir', type=str, default="./data/test/predict/")
    parser.add_argument('-s', '--snapshot', type=str, default="./bce/2211/AttU_Nettt_97.pth")
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-nc', '--num_class', type=int, default=2)
    parser.add_argument('-gid', '--gpu_id', type=int, default=0)
    args = parser.parse_args()
    multi_predict(args)
