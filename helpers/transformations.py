'''
Created on 12 lis 2020

@author: spasz
'''
import cv2
import numpy as np
import datetime as dt
import logging


def Rotate(image, angle):
    ''' Rotate image by angle.'''
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    logging.debug('Rotation by %u.', angle)
    return result


def Flip(image):
    ''' Flips image horizontally.'''
    logging.debug('Flipped horizontaly.')
    return cv2.flip(image, 1)


def RandomlyTransform(image):
    ''' Use random transformation for image augmentation.'''
    from random import seed
    from random import randint
    seed(dt.datetime.utcnow())
    method = randint(0, 2)
    if (method == 0):
        return Rotate(image, angle=randint(5, 45))
    elif (method == 1):
        return Flip(image)
    else:
        return Flip(image)
