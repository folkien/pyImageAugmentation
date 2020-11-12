#!/usr/bin/python3
import os
import sys
from helpers.hashing import IsSha1Name
from helpers.files import GetFilename, RenameToSha1Filepath, GetNotExistingSha1Filepath, IsImageFile
from helpers.transformations import RandomlyTransform
import argparse
import logging
import cv2
from os import walk


# Arguments and config
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str,
                    required=False, help='Input path')
parser.add_argument('-ar', '--augmentation', action='store_true',
                    required=False, help='Process extra image augmentation.')
parser.add_argument('-v', '--verbose', action='store_true',
                    required=False, help='Show verbose finded and processed data')
args = parser.parse_args()

if (args.input is None):
    print('Error! No arguments!')
    sys.exit(-1)


# Enabled logging
if (__debug__ is True):
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
else:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logging.debug('Logging enabled!')


excludes = ['.', '..', './', '.directory']
f = []
for (dirpath, dirnames, filenames) in walk(args.input):
    for f in filenames:
        # Process only image files and not exlcudes
        if (f not in excludes) and (IsImageFile(f)):
            # Rename only files which has not SHA-1 name
            if (IsSha1Name(GetFilename(f)) == False):
                f = RenameToSha1Filepath(f, dirpath)

            # If enabled then augmentate data
            if (args.augmentation):
                image = cv2.imread(dirpath+f)
                image = RandomlyTransform(image)
                newFilename, newFilepath = GetNotExistingSha1Filepath(
                    f, dirpath)
                cv2.imwrite(newFilepath, image)
                logging.info('New augmented file %s.', dirpath+newFilepath)

    logging.debug('Number of files : %u.' % len(filenames))
