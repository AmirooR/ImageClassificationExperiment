import numpy as np
ds_root = '/Users/amirrahimi/Public/VOC2012/ImageSets/Main/'
f = open(ds_root+'aeroplane_val.txt','r')
lines = f.readlines()
npLines = np.asarray(lines)
l = [npLines[i][0:11] for i in np.arange(npLines.shape[0])]
gt = [int(npLines[i][12:14]) for i in np.arange(npLines.shape[0])]
aerNames = l
aerGT = gt
aerPosIdx = np.where( np.asarray(aerGT) == 1 )
npAerNames  = np.asarray(aerNames)
aerPosNames = npAerNames[aerPosIdx]
import os
aerNegNames = np.lib.arraysetops.setdiff1d(npAerNames,aerPosNames,True)
dst = '/Users/amirrahimi/Public/VOCVirtual/Aeroplane/VAL/POS/'
i = 1
ds_root = '/Users/amirrahimi/Public/VOC2012/JPEGImages/'
print 'Linking Positives'
for x in aerPosNames:
    print i
    os.system('ln -s '+ds_root+x+'.jpg '+dst+x+'.jpg')
    i = i + 1
dst = '/Users/amirrahimi/Public/VOCVirtual/Aeroplane/VAL/NEG/'
i = 1
print 'Linking Negatives'
for x in aerNegNames:
    print i
    os.system('ln -s '+ds_root+x+'.jpg '+dst+x+'.jpg')
    i = i + 1

