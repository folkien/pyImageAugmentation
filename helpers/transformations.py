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


def Translate(image, x, y):
    ''' Translates image.'''
    logging.debug('Translated %u,%u.', x, y)
    rows, cols, depth = image.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, M, (cols, rows))


def Affine(image):
    ''' image.'''


def Perspective(image, factor=0.1):
    ''' image.'''
    logging.debug('Perspective transform %2.2f.', factor)
    h, w, depth = image.shape
    pts1 = np.float32([[w*factor, h*factor], [w*(1-factor), h*factor],
                       [w*factor, h*(1-factor)], [w*(1-factor), h*(1-factor)]])
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (w, h))


def RandomlyTransform(image):
    ''' Use random transformation for image augmentation.'''
    from random import seed
    from random import randint
    from random import uniform
    seed(dt.datetime.utcnow())
    method = randint(0, 3)
    if (method == 0):
        return Rotate(image, angle=randint(5, 45))
    elif (method == 1):
        return Flip(image)
    elif (method == 2):
        return Translate(image, x=randint(5, 100), y=randint(5, 100))
    elif (method == 3):
        return Perspective(image, factor=uniform(0.05, 0.3))
    else:
        return Flip(image)
