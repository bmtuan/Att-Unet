import argparse, sys

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from transform import random_transform_generator
from dataset import SEMDataset
from modules import *
from save_history import *
from loss import masked_bce_loss, masked_dice_loss, masked_dbce_loss

# from model import UNet
from advance_model import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Multualy exclusive with --predict.', action='store_true')
parser.add_argument('--predict', help='Multualy exclusive with --train.', action='store_true')
parser.add_argument('--prob_thres', help='Use with --predict.', action='store_true')
parser.add_argument('--evaluate', help='Multualy exclusive with --train.', action='store_true')
parser.add_argument('--transform', help='', action='store_true')
parser.add_argument('--loss_fn', type=str, default="bce")
parser.add_argument('--snapshot', type=str)
parser.add_argument('--train_dir', type=str)
parser.add_argument('--val_dir', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--init_epoch', type=int, default=0)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--val_interval', type=int, default=5)
parser.add_argument('--save_interval', type=int, default=10)

args = parser.parse_args()


def train():
    # transform generator
    if args.transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
    else:
        transform_generator = None

    # create custome dataset
    train_dataset = SEMDataset(os.path.join(args.train_dir, "img"),
                               os.path.join(args.train_dir, "label"),
                               transform_generator=transform_generator)
    val_dataset = SEMDataset(os.path.join(args.val_dir, "img"),
                             os.path.join(args.val_dir, "label"))

    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=args.num_workers,
                                             batch_size=args.batch_size, shuffle=False)

    # Model
    model = UNet(in_channels=11, n_classes=2, depth=2, batch_norm=True, padding=True)
    # model = UNet(in_channels=11, n_classes=2)

    if args.snapshot:
        model = torch.load(args.snapshot)
    model = model.cuda()

    # Loss function
    if args.loss_fn == "bce":
        criterion = masked_bce_loss
    elif args.loss_fn == "dice":
        criterion = masked_dice_loss
    elif args.loss_fn == 'dbce':
        criterion = masked_dbce_loss
    else:
        RaiseValueError("%s loss function is not supported" % args.loss_fn)

    # Optimizerd
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    # Saving History to csv
    header = ['epoch', 'train_loss', 'val_loss', 'val_acc']

    save_dir = os.path.join(args.save_dir, args.loss_fn)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file_name = os.path.join(save_dir, "history.csv")
    for i in range(args.init_epoch, args.init_epoch + args.n_epoch):
        # train the model
        train_loss = train_model(model, train_loader, criterion, optimizer, scheduler)

        # validation every 5 epoch
        if (i + 1) % args.val_interval == 0:
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, metric=True)
            print('Epoch %d, Train loss: %.5f, Val loss: %.5f, Val acc: %.4f' % (i + 1, train_loss, val_loss, val_acc))

            values = [i + 1, train_loss, val_loss, val_acc]
            export_history(header, values, save_dir, save_file_name)

        if (i + 1) % args.save_interval == 0:  # save model every save_interval epoch
            save_models(model, save_dir, i + 1)


def evaluate():
    if args.snapshot is None:
        RaiseValueError("--snapshot must be provided!")

    val_dataset = SEMDataset(os.path.join(args.val_dir, "img"),
                             os.path.join(args.val_dir, "label"))
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             num_workers=args.num_workers,
                                             batch_size=1,
                                             shuffle=False)
    # model = UNet(in_channels=11, n_classes=2)
    model = torch.load(args.snapshot)
    model = model.cuda()

    score_model(model, val_loader)


def predict():
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = SEMDataset(os.path.join(args.val_dir, "img"),
                         os.path.join(args.val_dir, "label"),
                         transform_generator=None)
    loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=1, shuffle=False)

    # model = UNet(in_channels=11, n_classes=2)
    model = torch.load(args.snapshot)
    model = model.cuda()

    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.cuda()

            probs = model.forward(images).data.cpu().numpy()  # 1 * C * H * W
            preds = np.argmax(probs, axis=1).astype(np.uint8) + 1  # 1 * H * W
            probs = np.max(probs, axis=1)  # 1 * H * W
            if args.prob_thres:
                high_prob_masks = (probs > 0.90).astype(np.uint8)
                preds = preds * high_prob_masks  # 1 * H * W
            pred = preds[0, ...]  # H x W

            no_value_mask = dataset.get_mask(batch_idx)  # H x W

            pred = pred * no_value_mask
            label = Image.fromarray(pred).convert("L")

            basename = dataset.get_basename(batch_idx)
            label.save(os.path.join(args.save_dir, "%s.png" % basename))


if __name__ == "__main__":
    if args.train:
        train()
    elif args.predict:
        predict()
    elif args.evaluate:
        evaluate()
    else:
        print("Please chose --train, --predict, --evaluate. Mutualy exclusive!")
