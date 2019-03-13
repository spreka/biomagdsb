import os
import sys
import glob
import scipy.misc
import numpy as np

imgdir = sys.argv[1]
maskdir = sys.argv[2]

imgnames = [os.path.basename(x) for x in glob.glob(os.path.join(imgdir,'*.png'))]
for imgname in imgnames:
    imgid = imgname[:-4]
    lfh = open(os.path.join(imgdir,imgid+'.txt'),'w')
    masks = glob.glob(os.path.join(maskdir,imgid,'masks','*.png'))
    for mask in masks:
        img = scipy.misc.imread(mask)
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        lfh.write("cell 0 0 0 %d %d %d %d 0 0 0 0 0 0 0\n"%(cmin,rmin,cmax,rmax))
    lfh.close()
