'''
Created on 12 lis 2020

@author: spasz
'''


def GetHexList():
    ''' Returns hex list.'''
    return [ '0','1','2','3','4','5','6','7','8','9',
            'a','b','c','d','e','f']
    
def IsSha1Name(name):
    ''' Check if filename is a SHA-1 name.''' 
    result = True
    
    if (len(name) == 40):
        for letter in name:
            if letter.lower() not in GetHexList(): 
                result = False
                break
    else:
        result= False
                
    return result

counter = 0
def GetRandomSha1():
    '''Create image name'''
    import hashlib
    import datetime
    global counter
    m = hashlib.sha1()
    m.update(str(counter).encode('ASCII'))
    m.update(str(datetime.datetime.now().timestamp()).encode('ASCII'))
    counter+=1
    return m.hexdigest()