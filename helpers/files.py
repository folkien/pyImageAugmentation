'''
Created on 10 wrz 2020

@author: spasz
'''

from pathlib import Path


def GetFilename(path):
    ''' Returns filename without extension'''
    import os
    return os.path.splitext(path)[0]


def GetExtension(path):
    ''' Returns extension'''
    import os
    return os.path.splitext(path)[1]

def CreateOutputDirectory(filepath):
    # Create output path
    objectDirectory = 'output'
    path = '%s/%s' % (objectDirectory, filepath)
    Path(path).mkdir(parents=True, exist_ok=True)
    
def GetNewShaFilepath():
    ''' Returns new SHA-1 Filepath.'''
    
