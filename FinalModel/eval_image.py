#!/usr/bin/python
import os
import os.path
import numpy as np
import skimage.io
import sys
import json
from skimage.measure import regionprops


gold_extension = ".tiff"

def computeIOU(gtIndices=None, sgIndices=None):
    return len(np.intersect1d(gtIndices,sgIndices,assume_unique=True))/len(np.union1d(gtIndices,sgIndices))


def evalImage(goldMask=None, segmMask=None):
    precision = 0.0
    thresholdLevels = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    nofThresh = 10

    width = goldMask.shape[1]
    height = goldMask.shape[0]
    
    pred_width = segmMask.shape[1]
    pred_height = segmMask.shape[0]

    if width != pred_width:
      return precision
        
    if height != pred_height:
      return precision
    
    uvGT,uvGTidx = np.unique(goldMask,return_inverse=True)
    uvGT = np.delete(uvGT, np.where(uvGT==0))
    rgoldMask = np.reshape(uvGTidx,goldMask.shape)
    uvPM,uvPMidx = np.unique(segmMask,return_inverse=True)
    uvPM = np.delete(uvPM, np.where(uvPM == 0))
    rsegmMask = np.reshape(uvPMidx,segmMask.shape)

    nofGT = len(uvGT)
    nofAllPM = len(uvPM)

    TPs = np.zeros((nofThresh, nofGT))

    gtProps = regionprops(rgoldMask)
    pmProps = regionprops(rsegmMask)

    for gtInd in range(nofGT):
        coords = gtProps[gtInd].coords
        overlappingPM = rsegmMask[coords[:,0],coords[:,1]]

        uvOverlappingPM = np.unique(overlappingPM)
        uvOverlappingPM = np.delete(uvOverlappingPM, np.where(uvOverlappingPM == 0))
        nofPM = len(uvOverlappingPM)

        for pmInd in range (nofPM):
            iou = computeIOU(np.ravel_multi_index(np.transpose(coords), goldMask.shape), \
                              np.ravel_multi_index(np.transpose(pmProps[uvOverlappingPM[pmInd-1]-1].coords),goldMask.shape))
            for tInd in range(nofThresh):
                if iou>thresholdLevels[tInd]:
                    TPs[tInd,gtInd] = 1

    TP = np.sum(TPs, axis=1)
    FN = nofGT - TP
    FP = nofAllPM - TP
    precision = np.mean(TP / (TP+FP+FN))
    return precision

    
def eval_dir(gold_dir, segm_dir):
  values = []
  for file in os.listdir(segm_dir):
     if file.endswith(".tiff") or  file.endswith(".png"):
       basename = os.path.splitext(os.path.basename(file))[0]
       goldfile = basename + gold_extension
       gold_path = os.path.join(gold_dir,goldfile)
       segm_path = os.path.join(segm_dir,file)
       gold = skimage.io.imread(gold_path)
       segm = skimage.io.imread(segm_path)
       value = evalImage(gold, segm)
       values.append(value)
       print(basename + '\t' + "{:1.3f}".format(value))

  valarr = np.array(values)
  print("Evaluation of", segm_dir)
  print("  count:", len(values))
  print("  avg: ", np.average(valarr))
  print("  dev: ", np.std(valarr))
  print("  min: ", np.min(valarr))
  print("  max: ", np.max(valarr))



jsn = json.load(open(sys.argv[1]))
eval_dir(os.path.join(os.curdir, jsn["eval_params"]["gold_dir"],"masks"),
         os.path.join(os.curdir, jsn["eval_params"]["result_dir"]))



