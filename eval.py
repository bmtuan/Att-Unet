import argparse, sys

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

from dataset import SEMDataset
from modules import *
from save_history import *
from advance_model import AttU_Net
import params


def eval(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = SEMDataset(
        os.path.join(args.input_dir, "imgs"),
        os.path.join(args.input_dir, "labels"),
        num_class=args.num_class,
    )
    loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=1, shuffle=False)

    # init model
    model = AttU_Net(in_channels=3, n_classes=args.num_class)
    model = model.cuda()

    load_checkpoint(args.snapshot, model, None)

    score_model(model, loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-vd', '--val_dir', type=str,
    #                     default='./data/val/')
    parser.add_argument('-id', '--input_dir', type=str)
    parser.add_argument('-s', '--snapshot', type=str, default="./dice/dice/AttU_Nettt_final.pth")
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-nc', '--num_class', type=int, default=3)
    args = parser.parse_args()

    eval(args)
