#from post_processing import *
import numpy as np
from PIL import Image
import glob as gl
import numpy as np
from PIL import Image
import torch

def accuracy_check(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if 'str' in str(type(item)):
            item = np.array(Image.open(item))
        elif 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.numpy()
        np_ims.append(item)

    compare = np.equal(np_ims[0], np_ims[1])
    
    mask_ignore_background = np.ones(mask.shape)
    mask_ignore_background[mask == 0] = 0
    
    accuracy = np.sum(compare * mask_ignore_background)

    return accuracy / np.sum(mask_ignore_background)

def accuracy_check_for_batch(masks, predictions, batch_size):
    total_acc = 0
    for index in range(batch_size):
        total_acc += accuracy_check(masks[index], predictions[index])
    return total_acc / batch_size

# Experimenting
if __name__ == '__main__':
    pass