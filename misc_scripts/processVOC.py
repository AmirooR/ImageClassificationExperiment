# coding: utf-8
import numpy as np
get_ipython().system(u'ls -F ')
a = np.loadtxt('aeroplane_train.txt')
get_ipython().set_next_input(u'f = open');get_ipython().magic(u'pinfo open')
f = open('aeroplane_train.txt','r')
lines = f.readlines()
lines
lines[0]
lines[0][0:10]
lines[0][0:11]
lines[0][0:11]
lines[0][11:13]
lines[0][12:14]
int(lines[0][12:14])
int(' 1')
planeList = lines[:][0:11]
planeList
planeList = lines[0][0:11]
planeList
npLines = np.asarray(lines)
npLines
npLines[:][0:11]
npLines.shape
npLines[0][0:11]
l = [npLines[i][0:11] for i in arange(shape(npLines)[0])]
l = [npLines[i][0:11] for i in np.arange(shape(npLines)[0])]
l = [npLines[i][0:11] for i in np.arange(npLines.shape[0])]
l
l[0]
gt = [int(npLines[i][13:15]) for i in np.arange(npLines.shape[0])]
gt
npLines[0][13:15]
npLines[0][12:14]
gt = [int(npLines[i][12:14]) for i in np.arange(npLines.shape[0])]
gt
aerNames = l
aerGT = gt
f = open('person_train.txt','r')
lines = f.readlines()
npLines = np.asarray(lines)
l = [npLines[i][0:11] for i in np.arange(npLines.shape[0])]
gt = [int(npLines[i][12:14]) for i in np.arange(npLines.shape[0])]
perNames = l
perGT = gt
list(l)
l
get_ipython().magic(u'who ')
perNames
get_ipython().magic(u'who ')
aerGT
np.where( gt == 1)
gt
np.where( gt == -1)
get_ipython().magic(u'pinfo np.where')
np.where(gt > 0.5)
gt
gt[0]
gt[1]
np.where( gt == 1 )
np.where( gt == 1 )[0]
gt
npGT = np.asarray(gt)
np.where( gt == 1 )
gt
np.where( npGT == 1 )
get_ipython().magic(u'who ')
aerPosIdx = np.where( np.asarray(aerGT) == 1 )
aerPosIdx
aerPosIdx.shape
len(aerPosIdx)
aerPosIdx[0]
len(aerPosIdx[0])
get_ipython().magic(u'who ')
len(aerPosIdx[0])
aerZeroIdx = np.where( np.asarray(aerGT) == 0 )
len(aerZeroIdx[0])
aerZeroIdx
get_ipython().magic(u'who ')
aerPosIdx = np.where( np.asarray(aerGT) == 1 )
len(aerPosIdx[0])
perPosIdx = np.where(np.asarray(perGT) == 1)
len(perPosIdx[0])
aerNames[perPosIdx]
aerNames[perPosIdx[0]]
aerNames
npAerNames  = np.asarray(aerNames)
npAerNames[perPosIdx]
npAerNames[perPosIdx].shape
npPerNames = np.asarray(perNames)
get_ipython().magic(u'who ')
npAerNames[aerPosIdx]
npPerNames  = np.asarray(perNames)
npAerNames[perPosIdx]
npAerNames[aerPosIdx]
aerPosNames = npAerNames[aerPosIdx]
perPosNames = npPerNames[perPosIdx]
aerPosNames.shape
perPosNames.shape
set(perPosNames)
t = list( set(perPosNames) & set(aerPosNames) )
t
import os
os.listdir('/Users/amirrahimi/Public/VOC2012/JPEGImages')
a = os.listdir('/Users/amirrahimi/Public/VOC2012/JPEGImages')
a
a[0]
import cv,cv2
maxX ,maxY = 0
maxX ,maxY = 0,0
maxX
a[0][:-3]
a[0][-3:]
for i in a:
    if i[-3:] == 'jpg':
        get_ipython().set_next_input(u'        img = cv2.imread');get_ipython().magic(u'pinfo cv2.imread')
        
get_ipython().magic(u'pinfo cv2.imread')
root = '/Users/amirrahimi/Public/VOC2012/JPEGImages/'
for i in a:
    if i[-3:] == 'jpg':
        img = cv2.imread(root+i)
        x = img.shape[0]
        y = img.shape[1]
        if x > maxX: maxX = x
        if y > maxY: maxY = y
        
maxX
maxY
get_ipython().magic(u'who ')
npAerNames.shape
npPerNames.shape
perPosNames.shape
aerPosNames.shape
get_ipython().magic(u'pinfo set')
get_ipython().magic(u'pinfo np.intersect1d')
get_ipython().magic(u'pinfo np.lib.arraysetops')
get_ipython().system(u'ls -F ')
get_ipython().magic(u'pinfo np.lib.arraysetops.setdiff1d')
perNegNames = np.lib.arraysetops.setdiff1d(npPerNames,perPosNames,True)
perNegNames.shape
perPosNames.shape
perPosNames
root
dst = '/Users/amirrahimi/Public/VOCVirtual/Aeroplane/POS/'
for x in perPosNames:
    os.system('ln -s '+root+x+'.jpg '+dst+x+'.jpg')
    
dst = '/Users/amirrahimi/Public/VOCVirtual/Person/NEG'
dst = '/Users/amirrahimi/Public/VOCVirtual/Person/NEG/'
for x in perNegNames:
    os.system('ln -s '+root+x+'.jpg '+dst+x+'.jpg')
    
get_ipython().magic(u'pinfo %save')
get_ipython().magic(u'save /Users/amirrahimi/Desktop/processVOC 1-140')