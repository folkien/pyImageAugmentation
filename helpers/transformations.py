'''
Created on 12 lis 2020

@author: spasz
'''
import cv2
import numpy as np
import datetime as dt
import logging


def AddNoise(image, noise_typ='gauss', var=0.1):
    ''' Add noise to image.'''
    logging.info('Adding noise %2.2f!', var)
    if (noise_typ == 'gauss'):
        row, col, ch = image.shape
        mean = 0
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image*gauss
        return noisy
    elif noise_typ == 's&p':
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == 'poisson':
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == 'speckle':
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


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


def Mirror(image):
    ''' Mirror image horizontally.'''
    logging.debug('Mirroring horizontaly.')
    flipped = cv2.flip(image, 1)
    return cv2.hconcat([image, flipped])


def Mosaic(image):
    ''' Mosaic image.'''
    logging.debug('Mosaic.')
    h, w, depth = image.shape
    flipped = cv2.flip(image, 1)
    mosaic = cv2.hconcat([image, flipped])
    mosaic = cv2.vconcat([mosaic, mosaic])
    return cv2.resize(mosaic, (w, h), interpolation=cv2.INTER_AREA)


def Translate(image, x, y):
    ''' Translates image.'''
    logging.debug('Translated %u,%u.', x, y)
    rows, cols, depth = image.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, M, (cols, rows))


def Affine(image, factor=0.1):
    '''Affine transform image.'''
    from random import uniform
    logging.debug('Affine transform %2.2f.', factor)
    h, w, depth = image.shape
    pts1 = np.float32([[w*factor, h*factor], [w*(1-factor), h*factor],
                       [w*factor, h*(1-factor)]])
    pts2 = np.float32([[w*factor, h*factor], [w*(1-factor)+w*uniform(0, factor), h*factor+h*uniform(0, factor)],
                       [w*factor+w*uniform(0, factor), h*(1-factor)+h*uniform(0, factor)]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(image, M, (w, h))


def Perspective(image, factor=0.1):
    ''' image.'''
    logging.debug('Perspective transform %2.2f.', factor)
    h, w, depth = image.shape
    pts1 = np.float32([[w*factor, h*factor], [w*(1-factor), h*factor],
                       [w*factor, h*(1-factor)], [w*(1-factor), h*(1-factor)]])
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (w, h))


def Blur(image, size=10):
    '''Blur image.'''
    logging.debug('Blur %u.', size)
    # ksize
    ksize = (size, size)
    # Using cv2.blur() method
    return cv2.blur(image, ksize)


def Brightness(image, alpha=1):
    '''Brightness image.'''
    logging.debug('Brightness %2.2f.', alpha)
    return cv2.addWeighted(image, alpha, image, 0, 0)


def Contrast(image, alpha=1):
    '''Contrast image.'''
    logging.debug('Contrast %2.2f.', alpha)
    return cv2.addWeighted(image, alpha, image, 0, 127*(1-alpha))


def Saturation(image, factor=1):
    '''Saturation image.'''
    logging.debug('Saturation %2.2f.', factor)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = hsv[:, :, 1]*factor
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def Hue(image, factor=1):
    '''Hue image.'''
    logging.debug('Hue %2.2f.', factor)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = hsv[:, :, 0]*factor
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def RandomlyTransform(image):
    ''' Use random transformation for image augmentation.'''
    from random import seed
    from random import randint
    from random import uniform
    seed(dt.datetime.utcnow())

    method = randint(0, 12)
    if (method == 0):
        return Rotate(image, angle=randint(5, 45))
    elif (method == 1):
        return AddNoise(Flip(image), var=uniform(0.03, 0.15))
    elif (method == 2):
        return Translate(image, x=randint(5, 100), y=randint(5, 100))
    elif (method == 3):
        return Perspective(image, factor=uniform(0.05, 0.3))
    elif (method == 4):
        return Blur(Rotate(image, angle=randint(5, 45)), size=randint(3, 15))
    elif (method == 5):
        return Affine(image, factor=uniform(0.05, 0.3))
    elif (method == 6):
        return Mirror(image)
    elif (method == 7):
        return Brightness(image, alpha=uniform(0.5, 1.7))
    elif (method == 8):
        return Contrast(image, alpha=uniform(0.5, 1.9))
    elif (method == 9):
        return Mosaic(image)
    elif (method == 10):
        return Saturation(image, factor=uniform(0.5, 1.5))
    elif (method == 11):
        return Hue(image, factor=uniform(0.5, 1.5))
    else:
        return Flip(Rotate(image, angle=randint(5, 45)))
