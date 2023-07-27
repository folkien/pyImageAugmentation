#!/usr/bin/python3
import os
import sys
import random
from shutil import copyfile
from helpers.hashing import IsSha1Name
from helpers.files import GetFilename, RenameToSha1Filepath, GetNotExistingSha1Filepath, IsImageFile, CreateOutputDirectory, FixPath
from helpers.transformations import RandomlyTransform, Mosaic4,\
    RandomColorTransform, RandomShapeTransform, ResizeToWidth,\
    RandomDayWeatherTransform, TransformByName, ResizeLetterBox
from random import randint
import argparse
import logging
import cv2
from helpers.textAnnotations import ReadAnnotations,\
    ConvertAnnotationsToDetections
from helpers.boxes import ToAbsolute


# Arguments and config
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str,
                    required=True, help='Input path')
parser.add_argument('-o', '--output', type=str, nargs='?', const='generated', default='generated',
                    required=False, help='Output subdirectory name')
parser.add_argument('-ow', '--maxImageWidth', type=int, nargs='?', const=1280, default=1280,
                    required=False, help='Output image width / network width')
parser.add_argument('-oh', '--maxImageHeight', type=int, nargs='?', const=1280, default=1280,
                    required=False, help='Output image height / network height')
parser.add_argument('-an', '--augmentByName', nargs='+', type=str, default=[],
                    required=False, help='Augment annotations by name.')
parser.add_argument('-as', '--augumentShape', action='store_true',
                    required=False, help='Process extra image shape augmentation.')
parser.add_argument('-ac', '--augumentColor', action='store_true',
                    required=False, help='Process extra image color augmentation.')
parser.add_argument('-ad', '--augumentDayWeather', action='store_true',
                    required=False, help='Process augmentation day/weather.')
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

# Step 0.1 - random shuffle list (for big datasets it gives randomization)
random.shuffle(filenames)

# Step 1 - augment current images and make new
for f in filenames:
    # Set flag file is unmodified at beginning
    isModified = False
    # Read file text annotations if exists
    # and convert them to detections.
    annotations = ReadAnnotations(FixPath(dirpath)+f)
    detections = ConvertAnnotationsToDetections(annotations)

    # Rename only files which has not SHA-1 name
    if (IsSha1Name(GetFilename(f)) == False):
        # Old image name
        oldImageName = f
        # Rename image file
        newImageName = f = RenameToSha1Filepath(f, dirpath)
        # Optional annotations filename old/new
        oldTextFilename = GetFilename(oldImageName)+'.txt'
        # Check also annotations file if exists?
        if (os.path.exists(FixPath(dirpath)+oldTextFilename)):
            # Optional annotations filename old/new
            newTextFilename = GetFilename(newImageName)+'.txt'
            # Rename
            os.rename(FixPath(dirpath)+oldTextFilename,
                      FixPath(dirpath)+newTextFilename)

    # If enabled any augmentatation
    transform = None
    if (args.augumentShape) or (args.augumentColor) or (args.augumentDayWeather) or (args.augmentByName):
        # Read base image to memory
        image = cv2.imread(dirpath+f)
        # Recalculate bboxes in all detections to image pixel positions
        detections = [(label, conf, ToAbsolute(
            bbox, image.shape[1], image.shape[0])) for label, conf, bbox in detections]

        # Augmentate it
        # ----------------
        if (args.augumentDayWeather):
            image = RandomDayWeatherTransform(image, detections)
        if (args.augumentColor):
            transform = RandomColorTransform()
            image = transform.function(image)
        if (args.augumentShape):
            image = RandomShapeTransform(image, detections)
        if (args.augmentByName):
            for name in args.augmentByName:
                # Apply transform
                image = TransformByName(name, image, detections)

        # Set flag that file was modified
        isModified = True

    # If file is modified then save it
    if (isModified):
        # Create new filename
        newName, notused = GetNotExistingSha1Filepath(f, dirpath)
        if (transform is not None):
            newName = f"{transform.name}{newName}"
        outpath = dirpath+args.output+newName
        # Resize letterbox to network dimensions
        image = ResizeLetterBox(image, args.maxImageWidth, args.maxImageHeight)
        # Save
        cv2.imwrite(outpath, image)
        logging.info('New augmented file %s.', outpath)
        processedFiles += 1

        # Copy annotations if exists and copying enabled
        oldTextFilename = GetFilename(dirpath+f)+'.txt'
        if (os.path.exists(oldTextFilename)):
            # Optional annotations filename old/new
            newTextFilename = GetFilename(outpath)+'.txt'
            # Rename
            copyfile(oldTextFilename,
                     newTextFilename)


logging.debug('Processed files : %u.', processedFiles)
logging.debug('Number of files : %u.', len(filenames))
