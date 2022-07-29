import argparse
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary

from transform import random_transform_generator
from dataset import SEMDataset
from modules import *
from save_history import *
from loss import bce_loss, dice_loss, dbce_loss, new_loss
from advance_model import AttU_Net
from base_model import UNet
import params
import wandb
wandb.init()
def train(args):
    # transform generator
    if args.transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.95, 0.95),
            max_scaling=(1.05, 1.05),
            # flip_x_chance=0.1,
            # flip_y_chance=0.1,
        )
    else:
        transform_generator = None

    # create custome dataset
    train_dataset = SEMDataset(os.path.join(args.train_dir, "image"),
                               os.path.join(args.train_dir, "mask"),
                               num_class=args.num_class,
                               transform_generator=transform_generator)
    val_dataset = SEMDataset(os.path.join(args.val_dir, "image"),
                             os.path.join(args.val_dir, "mask"),
                             num_class=args.num_class)
    # Dataloader
    # print(train_dataset[0])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=args.num_workers, batch_size=1,
                                             shuffle=False)
    # Model
    #
    if args.model == "AttU_Nettt":
        print('AttU_Nettt')
        model = AttU_Net(in_channels=3, n_classes=args.num_class)
    else:
        # default is classical UNet
        print('Unet')
        model = UNet(in_channels=3, n_classes=args.num_class,
                     depth=5, batch_norm=True, padding=True)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, verbose=True)

    device = torch.device("cuda:%d" %
                          args.gpu_id if torch.cuda.is_available() else "cpu")
    # global PATH

    model = model.to(device)

    if args.pretrained:
        print(f'load checkpoint from {args.check_point}')
        load_checkpoint(args.check_point, model, optimizer, device=device)

    # Loss function
    if args.loss_fn == "bce":
        criterion = bce_loss
    elif args.loss_fn == "dice":
        criterion = dice_loss
    elif args.loss_fn == 'dbce':
        criterion = dbce_loss
    else:
        raise ValueError("%s loss function is not supported" % args.loss_fn)
    # criterion = new_loss
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    start_epoch = 0
    best_acc = 0
    
    if args.snapshot:
        start_epoch = load_checkpoint(args.snapshot, model, optimizer)

    # Saving History to csv
    header = ['epoch', 'train_loss', 'val_loss', 'val_acc']

    save_dir = os.path.join(args.save_dir, args.date)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        for epoch in tqdm(range(start_epoch, args.n_epoch)):
            # train the model
            train_loss = train_model(
                model, train_loader, criterion, optimizer, scheduler, device)
            train_loss_list.append(train_loss)
            # validation every args.val_interval epoch
            if (epoch + 1) % args.val_interval == 0:
                val_loss, val_acc = evaluate_model(
                    model, val_loader, criterion, device, metric=True)
                print('Epoch %d,Train loss: %.5f, Val loss: %.5f, Val acc: %.4f' % (
                    epoch + 1, train_loss, val_loss, val_acc))

                values = [epoch + 1, train_loss, val_loss, val_acc]
                export_history(header, values, save_dir,
                               os.path.join(save_dir, "history.csv"))
                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc)
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    })
            # save model every save_interval epoch
            
            # if (epoch + 1) % args.save_interval == 0:
            if val_acc > best_acc:
                save_checkpoint(os.path.join(save_dir, "best.pth"), model, optimizer,
                                epoch)
                best_acc = val_acc
        # plot_metrics(train_loss_list, val_loss_list, val_acc_list, save_dir,
        #              os.path.join(save_dir, "metrics.png"))
    except KeyboardInterrupt:
        save_checkpoint(os.path.join(save_dir, "{0}_final.pth".format(
            args.model)), model, optimizer, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--transform', help='',
                        action='store_true', default=True)
    parser.add_argument('-l', '--loss_fn', type=str, default="dice")
    parser.add_argument('-s', '--snapshot', type=str)
    parser.add_argument('-m', '--model', type=str, default="AttU_Nettt")
    parser.add_argument('-td', '--train_dir', type=str,
                        default='./data/train/')
    parser.add_argument('-vd', '--val_dir', type=str,
                        default='data/val/')
    parser.add_argument('-sd', '--save_dir', type=str, default='checkpoint/')
    parser.add_argument('-lr', '--lr', type=float, default=3e-5)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5)
    parser.add_argument('-ne', '--n_epoch', type=int, default=200)
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-vi', '--val_interval', type=int, default=1)
    parser.add_argument('-si', '--save_interval', type=int, default=2)
    parser.add_argument('-nc', '--num_class', type=int, default=2)
    parser.add_argument('-gid', '--gpu_id', type=int, default=0)
    parser.add_argument('-cp', '--check_point', type=str,
                        default='./checkpoint/20_12.pth')
    parser.add_argument('-pr', '--pretrained', type=bool, default=False)
    parser.add_argument('-dt', '--date', type=str, default="0707")
    args = parser.parse_args()
    train(args)
