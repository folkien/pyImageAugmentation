#!/usr/bin/python3
import os
import sys
import time
import argparse
import logging
from os import walk

def GetExtension(path):
    ''' Returns extension'''
    import os
    return os.path.splitext(path)[1]

counter = 0
def GetShaName():
    '''Create image name'''
    import hashlib
    import datetime
    global counter
    m = hashlib.sha1()
    m.update(str(counter).encode('ASCII'))
    m.update(str(datetime.datetime.now().timestamp()).encode('ASCII'))
    counter+=1
    return m.hexdigest()

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

    

f = []
for (dirpath, dirnames, filenames) in walk(args.input):
    for f in filenames:
        extension = GetExtension(f).lower()
        oldFilepath = dirpath+f
        newFilepath =  dirpath+GetShaName()+extension
        os.rename(oldFilepath, newFilepath)
        logging.debug('%s -> %s.' % (oldFilepath, newFilepath))
    logging.debug("Number of files : %u." % len(filenames))
