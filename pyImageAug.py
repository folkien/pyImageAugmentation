#!/usr/bin/python3
import os
import sys
from helpers.hashing import IsSha1Name
from helpers.files import GetFilename, RenameToSha1Filepath, GetNotExistingSha1Filepath, IsImageFile, CreateOutputDirectory
from helpers.transformations import RandomlyTransform, Mosaic4
from random import randint
import argparse
import logging
import cv2


# Arguments and config
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str,
                    required=True, help='Input path')
parser.add_argument('-o', '--output', type=str, nargs='?', const='', default='',
                    required=False, help='Output subdirectory name')
parser.add_argument('-ar', '--augmentation', action='store_true',
                    required=False, help='Process extra image augmentation.')
parser.add_argument('-v', '--verbose', action='store_true',
                    required=False, help='Show verbose finded and processed data')
args = parser.parse_args()

if (args.input is None):
    print('Error! No arguments!')
    sys.exit(-1)

# Creates output subdirectory
if (len(args.output)):
    CreateOutputDirectory(args.input+'/'+args.output)
    args.output = args.output+'/'

# Enabled logging
if (__debug__ is True):
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
else:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logging.debug('Logging enabled!')


excludes = ['.', '..', './', '.directory']
f = []
dirpath = args.input
processedFiles = 0
filenames = os.listdir(dirpath)

# Step 0 - filter only images
filenames = [f for f in filenames if (f not in excludes) and (IsImageFile(f))]
totalImages = len(filenames)

# Step 1 - augment current images and make new
for f in filenames:
    # Rename only files which has not SHA-1 name
    if (IsSha1Name(GetFilename(f)) == False):
        f = RenameToSha1Filepath(f, dirpath)

    # If enabled then augmentate data
    if (args.augmentation):
        image = cv2.imread(dirpath+f)
        image = RandomlyTransform(image)
        newName, notused = GetNotExistingSha1Filepath(
            f, dirpath)
        outpath = dirpath+args.output+newName
        cv2.imwrite(outpath, image)
        logging.info('New augmented file %s.', outpath)
        processedFiles += 1

# Step 2 - make mosaic images
if (len(filenames) >= 4):
    n = int(len(filenames)*0.3)
    for i in range(n):
        im1 = cv2.imread(dirpath+filenames[randint(0, totalImages-1)])
        im2 = cv2.imread(dirpath+filenames[randint(0, totalImages-1)])
        im3 = cv2.imread(dirpath+filenames[randint(0, totalImages-1)])
        im4 = cv2.imread(dirpath+filenames[randint(0, totalImages-1)])
        image = Mosaic4(im1, im2, im3, im4)
        newName, notused = GetNotExistingSha1Filepath(
            filenames[randint(0, totalImages-1)], dirpath)
        outpath = dirpath+args.output+newName
        cv2.imwrite(outpath, image)
        logging.info('New mosaic file %s.', outpath)
        processedFiles += 1


logging.debug('Processed files : %u.', processedFiles)
logging.debug('Number of files : %u.', len(filenames))
