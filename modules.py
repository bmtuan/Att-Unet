import os
import copy

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torchvision.utils import save_image

from sklearn.metrics import f1_score, confusion_matrix

from tqdm import tqdm


def train_model(model, data_loader, criterion, optimizer, scheduler, device):
    """Train the model and report validation error with training error
    Args:
        model: the model to be trained
        criterion: loss function
        train_loader (DataLoader): training dataset
    """
    model.train()
    epoch_loss = []

    pbar = tqdm(data_loader)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        # print(images.shape)
        # print(labels.shape)
        # print(outputs.shape)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("%.3f" % loss.item())
        epoch_loss.append(loss.item())
        # print(loss.item())
    scheduler.step(np.mean(epoch_loss))
    return np.mean(epoch_loss)


def evaluate_model(model, data_loader, criterion, device, metric=False):
    """
        Calculate loss over train set
    """
    total_loss = 0
    # i = 0
    all_acc = []
    # all_pixels  = []
    save = 'data/train/check_error2'
    model.eval()
    with torch.no_grad():
        for batch, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            outputs = outputs.cpu().numpy()
            predicted = np.argmax(outputs, axis=1)

            #####
            # predict = predicted.astype(np.uint8)[0, ...]
            # predicted_mask = Image.fromarray(predict * 100).convert('L')
            # predicted_mask.save(f'./data/val/predict/{i}.png')
            # print(f'predicted_mask {i} has been saved ')
            # i = i + 1
            #####

            labels = labels.cpu().numpy()
            labels = np.argmax(labels, axis=1)
            matches = (predicted == labels).astype(np.uint8)
            matches_zeros = ((predicted + labels) == 0).astype(np.uint8)

            matches_1 = ((predicted + labels) == 1).astype(np.uint8)
            matches_2 = ((predicted + labels) == 2).astype(np.uint8)

            acc = np.sum(matches_2) / (np.sum(matches_1) + np.sum(matches_2))
            # if acc < 0.8:
            #     print(acc, batch)

            #     cv2.imwrite(os.path.join(save, str(batch) + '_1.png'),
            #                 predicted.reshape(288, 512) * 255)
            #     save_image(images[0], os.path.join(
            #         save, str(batch) + '_3.png'))
            #     cv2.imwrite(os.path.join(save, str(batch) + '_2.png'),
            #                 labels.reshape(288, 512) * 255)

            if np.isnan(acc):
                all_acc.append(1)
            else:
                all_acc.append(acc)

                # all_acc.append(np.sum(matches) / (labels.shape[1] * labels.shape[2]))
                # all_acc.append(
                #     (np.sum(matches) - np.sum(matches_zeros)) / (labels.shape[1] * labels.shape[2] - np.sum(matches_zeros))
                # )

            # print('acc ', all_acc)
    return total_loss / len(data_loader), np.mean(np.array(all_acc))


def score_model(model, data_loader):
    """
        Calculate loss over train set
    """
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for batch, (images, labels) in enumerate(data_loader):
            images = images.cuda()  # 1 x C x H x W

            outputs = model(images)  # 1 x C x H x W
            outputs = outputs.cpu().numpy()
            preds = np.argmax(outputs, axis=1)  # 1 x H x W

            labels = labels.numpy()  # 1 x C x H x W
            masks = np.max(labels, axis=1)  # 1 x H x W
            labels = np.argmax(labels, axis=1)  # 1 x H x W

            pred = preds[0, ...]  # H x W
            mask = masks[0, ...]  # H x W
            label = labels[0, ...]  # H x W

            indices_y, indices_x = np.where(mask > 0)
            for y, x in zip(indices_y, indices_x):
                y_pred.append(pred[y, x])
                y_true.append(label[y, x])

    print("macro_f1", f1_score(y_true, y_pred, average='macro'))
    print("weighted_f1", f1_score(y_true, y_pred, average='weighted'))
    print(confusion_matrix(y_true, y_pred))
