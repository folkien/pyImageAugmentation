'''
Created on 12 lis 2020

@author: spasz
'''

from helpers.hashing import IsSha1Name,GetRandomSha1

def hashing_test():
    ''' Test '''
    assert(IsSha1Name('testsef.png') == False)
    assert(IsSha1Name('5e734393c78ff0d86f496a764a4928616fa71f50') == True)
    assert(IsSha1Name(GetRandomSha1()) == True)
    