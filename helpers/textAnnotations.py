'''
Created on 16 lis 2020

@author: spasz
'''
import os
from helpers.files import GetFilename, DeleteFile
import helpers.boxes as boxes


def __getAnnotationFilepath(imagePath):
    ''' Returns annotation filepath.'''
    return GetFilename(imagePath) + '.txt'


def GetImageFilepath(annotationPath):
    ''' Returns image filepath.'''
    path = GetFilename(annotationPath)
    for suffix in ['.png', '.jpeg', '.jpg', '.PNG', '.JPEG', '.JPG']:
        if (os.path.isfile(path+suffix) and os.access(path+suffix, os.R_OK)):
            return path+suffix

    return None


def IsExistsAnnotations(imagePath):
    ''' True if exists annotations file.'''
    path = __getAnnotationFilepath(imagePath)
    return os.path.isfile(path) and os.access(path, os.R_OK)


def IsExistsImage(annotationPath):
    ''' True if exists annotations file.'''
    path = GetFilename(annotationPath)
    for suffix in ['.png', '.jpeg', '.jpg', '.PNG', '.JPEG', '.JPG']:
        if (os.path.isfile(path+suffix) and os.access(path+suffix, os.R_OK)):
            return True

    return False


def ConvertAnnotationsToDetections(annotations):
    ''' Annotations convert to YOLO format.'''
    return [(classNumber, 100, box) for classNumber, box in annotations]


def ReadAnnotations(imagePath):
    '''Read annotations from file.'''
    annotations = []
    path = __getAnnotationFilepath(imagePath)
    if (os.path.exists(path)):
        with open(path, 'r') as f:
            for line in f:
                txtAnnote = (line.rstrip('\n').split(' '))
                classNumber = int(txtAnnote[0])
                box = (float(txtAnnote[1]), float(txtAnnote[2]),
                       float(txtAnnote[3]), float(txtAnnote[4]))
                box = boxes.Bbox2Rect(box)
                annotations.append((classNumber, box))

    return annotations


def DeleteAnnotations(imagePath):
    '''Delete annotations file.'''
    path = __getAnnotationFilepath(imagePath)
    DeleteFile(path)


def SaveAnnotations(imagePath, annotations):
    '''Save annotations for file.'''
    path = __getAnnotationFilepath(imagePath)
    with open(path, 'w') as f:
        for element in annotations:
            classNumber, box = element
            box = boxes.Rect2Bbox(box)
            f.write('%u %2.6f %2.6f %2.6f %2.6f\n' %
                    (classNumber, box[0], box[1], box[2], box[3]))
