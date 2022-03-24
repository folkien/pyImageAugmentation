'''
Created on 12 lis 2020

@author: spasz
'''
import cv2
import numpy as np
import datetime as dt
import logging
from random import randint, uniform


def GetContainingBox(box1, box2):
    ''' Get box containing box1 and box2.'''
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    # left top corner
    xl = min(x1, x2, x3, x4)
    yl = min(y1, y2, y3, y4)
    # right bottom corner
    xr = max(x1, x2, x3, x4)
    yr = max(y1, y2, y3, y4)
    # result
    return (xl, yl, xr, yr)


def GetArea(box):
    ''' Get Area of box'''
    x, y, x2, y2 = box
    return abs((x2-x)*(y2-y))

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


def Erode(image, size=10):
    '''Denoise image.'''
    logging.debug('Eroding %u.', size)
    return cv2.erode(image, (size, size))


def Dilate(image, size=10):
    '''Dilate image.'''
    logging.debug('Dilating %u.', size)
    return cv2.dilate(image, (size, size))


def SegmentationKmeans(image, K=3):
    ''' Segmentation image with kmeans.'''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    twoDimage = image.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)
    ret, label, center = cv2.kmeans(
        twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((image.shape))


def SegmentationContour(image):
    ''' Segmentation image with kmeans.'''
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST,
                                  cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    mask = np.zeros((256, 256), np.uint8)
    masked = cv2.drawContours(mask, [cnt], -1, 255, -1)
    dst = cv2.bitwise_and(image, image, mask=mask)
    segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    return segmented


def SegmentationThreshold(image):
    ''' Segmentation image with threshold.'''
    from skimage.filters import threshold_otsu
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    def filter_image(image, mask):
        r = image[:, :, 0] * mask
        g = image[:, :, 1] * mask
        b = image[:, :, 2] * mask
        return np.dstack([r, g, b])

    thresh = threshold_otsu(img_gray)
    img_otsu = img_gray < thresh
    filtered = filter_image(image, img_otsu)
    return filtered


def SegmentationFelzenszwalb(image):
    ''' Segmentation image with threshold.'''
    from skimage.segmentation import felzenszwalb
    from skimage.segmentation import mark_boundaries
    segments_fz = felzenszwalb(image,
                               scale=100,
                               sigma=0.5,
                               min_size=50)
    return mark_boundaries(image, segments_fz)


def SegmentationFeaturesClassifier(image):
    ''' '''
    from skimage import data, segmentation, feature, future
    from sklearn.ensemble import RandomForestClassifier
    from functools import partial
    # Build an array of labels for training the segmentation.
    # Here we use rectangles but visualization libraries such as plotly
    # (and napari?) can be used to draw a mask on the image.
    training_labels = np.zeros(image.shape[:2], dtype=np.uint8)
    training_labels[:130] = 1
    training_labels[:170, :400] = 1
    training_labels[600:900, 200:650] = 2
    training_labels[330:430, 210:320] = 3
    training_labels[260:340, 60:170] = 4
    training_labels[150:200, 720:860] = 4

    sigma_min = 1
    sigma_max = 16
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=False, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            multichannel=True)
    features = features_func(image)
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                 max_depth=10, max_samples=0.05)
    clf = future.fit_segmenter(training_labels, features, clf)
    result = future.predict_segmenter(features, clf)
    return result


def SelectiveSearch(image):
    ''' Selective search algorithm.'''
    # TODO
    # initialize OpenCV's selective search implementation and set the
    # input image
    rects = []
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    ss.process(rects)
    # loop over the current subset of region proposals
    for (x, y, w, h) in rects[:100]:
        # draw the region proposal bounding box on the image
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    return image


def SegmentationGrabCut(image):
    ''' Grab cut segmentation'''
    height, width = image.shape[:2]
    mask = np.zeros(image.shape[:2], dtype='uint8')
    rect = (1, 1, width-1, height-1)
    # allocate memory for two arrays that the GrabCut algorithm internally
    # uses when segmenting the foreground from the background
    fgModel = np.zeros((1, 65), dtype='float')
    bgModel = np.zeros((1, 65), dtype='float')
    # apply GrabCut using the the bounding box segmentation method
    (mask, bgModel, fgModel) = cv2.grabCut(image, mask, rect, bgModel,
                                           fgModel, iterCount=10, mode=cv2.GC_INIT_WITH_RECT)
    # we'll set all definite background and probable background pixels
    # to 0 while definite foreground and probable foreground pixels are
    # set to 1
    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
                          0, 1)
    # scale the mask from the range [0, 1] to [0, 255]
    outputMask = (outputMask * 255).astype('uint8')
    # apply a bitwise AND to the image using our mask generated by
    # GrabCut to generate our final output image
    output = cv2.bitwise_and(image, image, mask=outputMask)
    return output


def CornerDetection(image):
    ''' Segmentation image with threshold.'''
    from skimage.feature import corner_harris, corner_subpix, corner_peaks
    coords = corner_peaks(corner_harris(
        image), min_distance=5, threshold_rel=0.02)
    coords_subpix = corner_subpix(image, coords, window_size=13)
    # TODO


def MorphDilate(image, size=7):
    '''Morph Dilate image.'''
    logging.debug('Morph dilating %u.', size)
    # Create morph
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    # Apply to image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    bg = cv2.morphologyEx(lab[..., 0], cv2.MORPH_DILATE, se)
    lab[..., 0] = cv2.divide(lab[..., 0], bg, scale=255)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def Denoise(image, size=10):
    '''Denoise image.'''
    logging.debug('Denoising %u.', size)
    return cv2.fastNlMeansDenoisingColored(image, None, size, size, 7, 21)


def Blur(image, size=10):
    '''Blur image.'''
    logging.debug('Blur %u.', size)
    # ksize
    ksize = (size, size)
    # Using cv2.blur() method
    return cv2.blur(image, ksize)


def UnsharpMask(image,
                size=7,
                sigma=1,
                threshold=5,
                amount=1):
    ''' Apply unsharp mask to image.'''
    blurred = cv2.GaussianBlur(image, (size, size), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + 3.0, blurred, -3.0, 0)
    return sharpened


def UnsharpMask2(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    # For details on unsharp masking, see:
    # https://en.wikipedia.org/wiki/Unsharp_masking
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def BlurBox(image, box, size=10):
    '''Blurred rain on image.'''
    x1, y1, x2, y2 = box
    # kernel
    ksize = (size-1, size-1)
    # Using cv2.blur() method
    image[y1:y2, x1:x2] = cv2.blur(image[y1:y2, x1:x2], ksize)

    return image


def Brightness(image, alpha=1):
    '''Brightness image.'''
    logging.debug('Brightness %2.2f.', alpha)
    return cv2.addWeighted(image, alpha, image, 0, 0)


def Contrast(image, alpha=2):
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


def CLAHE(image, gridsize=8):
    ''' CLAHE (Contrast Limited Adaptive Histogram Equalization) .'''
    logging.debug('Clahe %u.', gridsize)
    # Create CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
    # Apply to image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


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


def Spotlight(image, alpha=1.2):
    ''' Adds vignette.'''
    import math
    logging.debug('Spotlight.')
    # Image properties
    h, w, depth = image.shape
    diag = math.sqrt(h*h+w*w)
    # Spotlight location and properties
    x = int(w/4) + randint(0, int(w/2))
    y = int(h/4) + randint(0, int(h/2))
    r = int(diag * uniform(0.20, 0.70))
    # Pattern matrix
    patterns = np.full([h, w, 3], 0, dtype=np.uint8)

    color = 255
    for r in range(0, r):
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

def GetResizedHeightToWidth(width, height, maxWidth=1280):
    ''' Returns resized values.'''
    if (width > maxWidth):
        ratio = maxWidth/width
        height = int(ratio*height)-1
        width = maxWidth

    return width, height


def ResizeToWidth(image, maxWidth=1024):
    ''' Resize image with handling aspect ratio.'''
    height, width = image.shape[:2]
    # Resize only if broader than max width
    if (width > maxWidth):
        width, height = GetResizedHeightToWidth(width, height, maxWidth)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Otherwise return original image
    return image


def FrameDetections(image, detections, minAreaRatio=0.30):
    ''' Frame to annotations.'''
    # Detections is none or empty
    if (detections is None) or (len(detections) == 0):
        return image

    # Get first bbox
    frameBbox = detections[0][2]
    # Calculate containing box
    for _label, _confidence, bbox in detections[1:]:
        frameBbox = GetContainingBox(frameBbox, bbox)

    # Check if area of containing bbox is too small?
    areaRatio = GetArea(frameBbox) / (image.shape[0]*image.shape[1])
    # If area ratio is to small, then fix it
    if (areaRatio < minAreaRatio):
        logging.warning(
            '(FrameDetections) Too low area ratio %2.2f!', areaRatio)
        width, height = image.shape[0:2]
        helpingBox = int(width*0.33), int(height *
                                          0.33), int(width*0.67), int(height*0.67)
        frameBbox = GetContainingBox(frameBbox, helpingBox)

    # Framing
    x1, y1, x2, y2 = frameBbox
    image = image[y1:y2, x1:x2]

    return image


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


def ComputeSkew(src_img):
    ''' Compuer image skew.'''
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')

    img = cv2.medianBlur(src_img, 3)

    edges = cv2.Canny(img,  threshold1=30,  threshold2=100,
                      apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30,
                            minLineLength=w / 4.0, maxLineGap=h/4.0)
    angle = 0.0

    # print(nlines)
    cnt = 0
    if (lines is not None):
        for x1, y1, x2, y2 in lines[0]:
            ang = np.arctan2(y2 - y1, x2 - x1)
            # print(ang)
            if math.fabs(ang) <= 30:  # excluding extreme rotations
                angle += ang
                cnt += 1

    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi


def Deskew(image):
    ''' Deskew image of plates.'''
    return Rotate(image, ComputeSkew(image))
#''' --------------------------------------'''


def RandomShapeTransform(image, detections):
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


def RandomColorTransform(image, detections):
    ''' Use random transformation for image augmentation.'''
    from random import seed
    from random import randint
    from random import uniform
    seed(dt.datetime.utcnow())
    # Addd color transformation
    method = randint(0, 11)
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
    elif (method == 11):
        image = Spotlight(image)

    return image


def RandomDayWeatherTransform(image, detections):
    ''' Use random transformation for image augmentation.'''
    from random import seed
    from random import randint
    from random import uniform
    seed(dt.datetime.utcnow())

    # Addd color transformation
    method = randint(0, 11)
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
        image = Rain(image)
    elif (method == 7):
        image = Vignette(image)
    elif (method == 8):
        image = Night(image)
    elif (method == 9):
        image = Spotlight(image)

    return image


def RandomlyTransform(image, detections):
    ''' Use random transformation for image augmentation.'''
    # Addd shape transformation
    image = RandomShapeTransform(image, detections)

    # Addd color transformation
    image = RandomColorTransform(image, detections)

    return image


def TransformByName(name, image, detections):
    '''
        Apply transformation opencv function
        based on transformation name.
    '''
    if (name == 'blur'):
        return Blur(image)
    if (name == 'contrast'):
        return Contrast(image)
    if (name == 'clahe'):
        return CLAHE(image)
    if (name == 'denoise'):
        return Denoise(image)
    if (name == 'dilate'):
        return Dilate(image)
    if (name == 'deskew'):
        return Deskew(image)
    if (name == 'erode'):
        return Erode(image)
    if (name == 'morphdilate'):
        return MorphDilate(image)
    if (name == 'noise'):
        return AddNoise(image)
    if (name == 'night'):
        return Night(image)
    if (name == 'unsharp'):
        return UnsharpMask(image, size=7, sigma=5)
    if (name == 'segkmeans'):
        return SegmentationKmeans(image)
    if (name == 'segcontuour'):
        return SegmentationContour(image)
    if (name == 'segth'):
        return SegmentationThreshold(image)
    if (name == 'segfsz'):
        return SegmentationFelzenszwalb(image)
    if (name == 'selsearch'):
        return SelectiveSearch(image)
    if (name == 'grabcut'):
        return SegmentationGrabCut(image)
    if (name == 'framedets'):
        return FrameDetections(image, detections)

    logging.warning('Unknown transformation %s.', name)
    return image
