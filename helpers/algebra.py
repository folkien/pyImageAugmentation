'''
Created on 1 paź 2020

@author: spasz
'''

import math
from numba import njit, float64, float64, boolean
from numba.types import UniTuple


@njit(cache=True)
def GetDistance(p1, p2):
    # Deprecated
    ''' Calculates euclidean distance between points.'''
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


@njit(cache=True)
def GetTranslation(p1, p2):
    ''' Calculates vector translation.'''
    x1, y1 = p1
    x2, y2 = p2
    return (x2-x1, y2-y1)


@njit(cache=True)
def GetMiddlePoint(p1, p2):
    ''' Calculates middle point.'''
    x1, y1 = p1
    x2, y2 = p2
    return ((x2+x1)/2, (y2+y1)/2)


@njit(float64(UniTuple(float64, 2), UniTuple(float64, 2)), cache=True)
def EuclideanDistance(p1, p2):
    ''' Calculates metric.'''
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


@njit(cache=True)
def RadiansToDegree(radian):
    ''' Returns degrees.'''
    return radian*180/math.pi


@njit(cache=True)
def ToRadian(degree):
    ''' Returns degrees.'''
    return degree*math.pi/180


@njit(cache=True)
def GetHypotenuse(a, b):
    ''' Returns hypotenuse of triangle a,b,c.'''
    return math.sqrt(a**2 + b**2)
