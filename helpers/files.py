'''
Created on 10 wrz 2020

@author: spasz
'''

from pathlib import Path
import os 
import logging
from helpers.hashing import GetRandomSha1


def GetFilename(path):
    ''' Returns filename without extension'''
    return os.path.splitext(path)[0]


def GetExtension(path):
    ''' Returns extension'''
    return os.path.splitext(path)[1]

def CreateOutputDirectory(filepath):
    # Create output path
    objectDirectory = 'output'
    path = '%s/%s' % (objectDirectory, filepath)
    Path(path).mkdir(parents=True, exist_ok=True)
    
def IsImageFile(filepath):
    ''' Checks if file is image file.'''
    return GetExtension(filepath).lower() in [ '.gif', '.png', '.jpg', '.jpeg', '.tiff']
    
def RenameToSha1Filepath(filename,dirpath):
    ''' Returns new SHA-1 Filepath.'''
    extension = GetExtension(filename).lower()
    newFilepath = oldFilepath = dirpath+filename

    # Try random hash until find not existsing file
    while (os.path.isfile(newFilepath) and os.access(newFilepath, os.R_OK)):
        newFilepath =  dirpath+GetRandomSha1()+extension
    
    os.rename(oldFilepath, newFilepath)
    logging.debug('%s -> %s.' % (oldFilepath, newFilepath))
    
