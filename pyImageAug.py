#!/usr/bin/python3
import os
import sys
import time
from helpers.hashing import IsSha1Name
from helpers.files import GetFilename,RenameToSha1Filepath
import argparse
import logging
from os import walk


# Arguments and config
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str,
                    required=False, help='Input path')
parser.add_argument('-v', '--verbose', action='store_true',
                    required=False, help='Show verbose finded and processed data')
args = parser.parse_args()

if (args.input is None):
    print("Error! No arguments!")
    sys.exit(-1)


# Enabled logging
if (__debug__ is True):
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
else:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logging.debug('Logging enabled!')


    

excludes = [ '.', '..', './', '.directory']
f = []
for (dirpath, dirnames, filenames) in walk(args.input):
    for f in filenames:
        # Rename only files which has not SHA-1 name
        if (f not in excludes) and (IsSha1Name(GetFilename(f)) == False): 
            RenameToSha1Filepath(f, dirpath)
            
    logging.debug("Number of files : %u." % len(filenames))
