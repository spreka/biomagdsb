import sys
import random
import os
import glob
import shutil

ratioval = 0.25
indir = sys.argv[1]
traindir = os.path.join(indir,'train')
valdir = os.path.join(indir,'val')
if not os.path.exists(traindir):
    os.mkdir(traindir)
if not os.path.exists(valdir):
    os.mkdir(valdir)

files = glob.glob(os.path.join(indir,'*.png'))
valimgnum = int(len(files) * ratioval)
for i in range(valimgnum):
    imgname = random.choice(files)
    files.remove(imgname)
    shutil.copy(imgname,os.path.join(valdir,os.path.basename(imgname)))

for fn in files:
    shutil.copy(fn,os.path.join(traindir,os.path.basename(fn)))
