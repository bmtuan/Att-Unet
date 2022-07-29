import cv2
import os


def get_frame(time):
    cam.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
    ret, frame = cam.read()
    if ret:
        name = './frame/frame_' + str(time) + 's.png'
        print('Creating...' + name)

        cv2.imwrite(name, frame)
    return ret


if __name__ == '__main__':
    INPUT_PATH = '/home/bmtuan/Desktop/data_aims/data_aims/'
    VIDEO_NAME = '008L正常.mp4'

    if not os.path.exists('frame'):
        os.makedirs('frame')

    cam = cv2.VideoCapture(INPUT_PATH + VIDEO_NAME)
    sec = 0
    frameRate = 0.1
    success = get_frame(sec)
    while success:
        sec = sec + frameRate
        sec = round(sec, 2)
        success = get_frame(sec)
