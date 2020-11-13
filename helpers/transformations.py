'''
Created on 12 lis 2020

@author: spasz
'''
import cv2
import numpy as np
import datetime as dt
import logging

#''' Color transformations.'''
#''' --------------------------------------'''


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


def AddPattern(image, alpha=1.0, columns=10, dotwidth=5):
    ''' Adds dot pattern to image.'''
    logging.debug('Adding pattern %2.2f,%u,%u.', alpha, columns, dotwidth)
    h, w, depth = image.shape
    patterns = np.full([h, w, 3], 0, dtype=np.uint8)
    step = w/columns

    for x in range(columns):
        x1 = int(x*step - dotwidth)
        x2 = int(x*step + dotwidth)
        for y in range(columns):
            y1 = int(y*step - dotwidth)
            y2 = int(y*step + dotwidth)
            cv2.rectangle(patterns, (x1, y1), (x2, y2), [255, 255, 255], -1)

    return cv2.addWeighted(image, 1, patterns, alpha, 0)


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


def Rain(image):
    '''Blurred rain on image.'''
    from random import randint
    logging.debug('Rain')
    h, w, depth = image.shape
    drops = randint(60, 100)
    for i in range(drops):
        size = randint(16, 40)
        x = randint(size+1, w-size-1)
        y = randint(size+1, h-size-1)
        # kernel
        ksize = (size-1, size-1)
        # Using cv2.blur() method
        image[y-size:y+size, x-size:x +
              size] = cv2.blur(image[y-size:y+size, x-size:x+size], ksize)

    return image


def Vignette(image, alpha=1.2):
    ''' Adds vignette.'''
    import math
    logging.debug('Vignette.')
    h, w, depth = image.shape
    diag = math.sqrt(h*h+w*w)
    x = int(w/2)
    y = int(h/2)
    patterns = np.full([h, w, 3], 0, dtype=np.uint8)

    color = 255
    for r in reversed(range(int(diag*0.15), int(diag*0.5))):
        # Draw a circle with blue line borders of thickness of 2 px
        patterns = cv2.circle(patterns, (x, y), r, [color, color, color], 2)
        color -= 1

    return cv2.addWeighted(image, 1, patterns, alpha, 0)


def __adjust_gamma(image, gamma=1.0):
    ''' Method for change gamma on picture.'''
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) *
                      255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)


def Night(image, desaturate=0.1, contrast=1.2, darkness=0.5):
    '''Night effect image.'''
    logging.debug('Night .')
    # Desaturate
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = hsv[:, :, 1]*desaturate
    # Increase contrast
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    image = cv2.addWeighted(image, contrast, image, 0, 0)
    # Reduce brightness
    return __adjust_gamma(image, 0.25)


def BlackBoxing(image, rows=6, cols=6):
    ''' Randomly inserts blackout boxes.'''
    from random import randint
    logging.debug('Blackboxing.')
    h, w = image.shape[0:2]
    # Box dimensions
    a = w/cols
    b = h/rows

    for x in range(rows):
        x1 = int(x*a)
        x2 = int((x+1)*a)
        for y in range(cols):
            y1 = int(y*b)
            y2 = int((y+1)*b)
            if (randint(0, 10) > 7):
                image = cv2.rectangle(image, (x1, y1), (x2, y2), [0, 0, 0], -1)

    return image


#''' Shape transformations.'''
#''' --------------------------------------'''


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


def Mosaic4(im1, im2, im3, im4):
    ''' Mosaic image.'''
    logging.debug('Mosaic.')
    h1, w1 = im1.shape[0:2]
    h2, w2 = im2.shape[0:2]
    h3, w3 = im3.shape[0:2]
    h4, w4 = im4.shape[0:2]
    # Create top
    h = max(h1, h2)
    im1 = cv2.resize(im1, (w1, h), interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(im2, (w2, h), interpolation=cv2.INTER_AREA)
    top = cv2.hconcat([im1, im2])
    # Create bottom
    h = max(h3, h4)
    im3 = cv2.resize(im3, (w3, h), interpolation=cv2.INTER_AREA)
    im4 = cv2.resize(im4, (w4, h), interpolation=cv2.INTER_AREA)
    bottom = cv2.hconcat([im3, im4])
    # Create mosaic
    ht, wt = top.shape[0:2]
    hb, wb = bottom.shape[0:2]
    w = max(wt, wb)
    top = cv2.resize(top, (w, ht), interpolation=cv2.INTER_AREA)
    bottom = cv2.resize(bottom, (w, hb), interpolation=cv2.INTER_AREA)
    mosaic = cv2.vconcat([top, bottom])
    return cv2.resize(mosaic, (w1, h1), interpolation=cv2.INTER_AREA)


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


#''' --------------------------------------'''

def RandomShapeTransform(image):
    ''' Use random transformation for image augmentation.'''
    from random import seed
    from random import randint
    from random import uniform
    seed(dt.datetime.utcnow())

    # Addd shape transformation
    method = randint(0, 8)
    if (method == 0):
        image = Rotate(image, angle=randint(5, 45))
    elif (method == 1):
        image = Flip(image)
    elif (method == 2):
        image = Flip(Rotate(image, angle=randint(5, 45)))
    elif (method == 3):
        image = Translate(image, x=randint(5, 100), y=randint(5, 100))
    elif (method == 4):
        image = Translate(Rotate(image, angle=randint(5, 45)),
                          x=randint(5, 100), y=randint(5, 100))
    elif (method == 5):
        image = Affine(image, factor=uniform(0.05, 0.3))
    elif (method == 6):
        image = Mirror(image)
    elif (method == 7):
        image = Perspective(image, factor=uniform(0.05, 0.3))
    elif (method == 8):
        image = Mosaic(image)

    return image


def RandomColorTransform(image):
    ''' Use random transformation for image augmentation.'''
    from random import seed
    from random import randint
    from random import uniform
    seed(dt.datetime.utcnow())
    # Addd color transformation
    method = randint(0, 10)
    if (method == 0):
        image = AddNoise(image, var=uniform(0.03, 0.15))
    elif (method == 1):
        image = Contrast(image, alpha=uniform(0.5, 1.9))
    elif (method == 2):
        image = Blur(image, size=randint(3, 15))
    elif (method == 3):
        image = Brightness(image, alpha=uniform(0.5, 1.7))
    elif (method == 4):
        image = Saturation(image, factor=uniform(0.5, 1.5))
    elif (method == 5):
        image = Hue(image, factor=uniform(0.5, 1.5))
    elif (method == 6):
        image = AddPattern(image, alpha=uniform(0.5, 0.9),
                           columns=randint(8, 16), dotwidth=randint(8, 40))
    elif (method == 7):
        image = Rain(image)
    elif (method == 8):
        image = Vignette(image)
    elif (method == 9):
        image = Night(image)
    elif (method == 10):
        image = BlackBoxing(image)

    return image


def RandomlyTransform(image):
    ''' Use random transformation for image augmentation.'''
    # Addd shape transformation
    image = RandomShapeTransform(image)

    # Addd color transformation
    image = RandomColorTransform(image)

    return image
