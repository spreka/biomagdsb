
#disable or enable this if you have oom problems
import tensorflow
config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.Session(config=config)


import os
import os.path
import mask_rcnn_additional
import kutils
import numpy
import cv2
import sys
import os
import skimage.morphology
import json
import utils
import visualize


class Segmentation:
    __mModel = None
    __mConfig = None
    __mModelDir = ""
    __mModelPath = ""
    __mLastMaxDim = mask_rcnn_additional.NucleiConfig().IMAGE_MAX_DIM
    __mConfidence = 0.5
    __NMSThreshold = 0.35

    '''
    @param pModelDir clustering Mask_RCNN model path
    '''
    def __init__(self, pModelPath, pConfidence=0.5, pNMSThreshold = 0.35, pMaxDetNum=512):
        if not os.path.isfile(pModelPath):
            raise ValueError("Invalid model path: " + pModelPath)

        self.__mConfidence = pConfidence
        self.__NMSThreshold = pNMSThreshold
        self.__mModelPath = pModelPath
        self.__mModelDir = os.path.dirname(pModelPath)
        self.__mMaxDetNum=pMaxDetNum

    def Segment(self, pImage, pPaddingRatio=0.0, pDilationSElem=None, pCavityFilling=False, pPredictSize=None):

        rebuild = self.__mModel is None

        if pPredictSize is not None:
            maxdim = pPredictSize
            temp = maxdim / 2 ** 6
            if temp != int(temp):
                maxdim = (int(temp) + 1) * 2 ** 6

            if maxdim != self.__mLastMaxDim:
                self.__mLastMaxDim = maxdim
                rebuild = True

        if rebuild:
            import model
            import keras.backend
            keras.backend.clear_session()
            print("Max dim changed (",str(self.__mLastMaxDim),"), rebuilding model")

            self.__mConfig = mask_rcnn_additional.NucleiConfig()
            self.__mConfig.DETECTION_MIN_CONFIDENCE = self.__mConfidence
            self.__mConfig.DETECTION_NMS_THRESHOLD = self.__NMSThreshold
            self.__mConfig.IMAGE_MAX_DIM = self.__mLastMaxDim
            self.__mConfig.IMAGE_MIN_DIM = self.__mLastMaxDim
            self.__mConfig.DETECTION_MAX_INSTANCES=self.__mMaxDetNum
            self.__mConfig.__init__()

            self.__mModel = model.MaskRCNN(mode="inference", config=self.__mConfig, model_dir=self.__mModelDir)
            self.__mModel.load_weights(self.__mModelPath, by_name=True)

        image = kutils.RCNNConvertInputImage(pImage)
        offsetX = 0
        offsetY = 0
        width = image.shape[1]
        height = image.shape[0]

        if pPaddingRatio > 0.0:
            image, (offsetX, offsetY) = kutils.PadImageR(image, pPaddingRatio)

        results = self.__mModel.detect([image], verbose=0)

        r = results[0]
        masks = r['masks']
        scores = r['scores']

        if masks.shape[0] != image.shape[0] or masks.shape[1] != image.shape[1]:
            print("Invalid prediction")
            return numpy.zeros((height, width), numpy.uint16), \
                   numpy.zeros((height, width, 0), numpy.uint8),\
                   numpy.zeros(0, numpy.float)


        count = masks.shape[2]
        if count < 1:
            return numpy.zeros((height, width), numpy.uint16), \
                   numpy.zeros((height, width, 0), numpy.uint8),\
                   numpy.zeros(0, numpy.float)

        if pPaddingRatio > 0.0:
            newMasks = numpy.zeros((height, width, count), numpy.uint8)

            for i in range(count):
                newMasks[:, :, i] = masks[offsetY: (offsetY + height), offsetX: (offsetX + width), i]

            masks = newMasks

        if pDilationSElem is not None:
            for i in range(count):
                masks[:, :, i] = cv2.dilate(masks[:, :, i], kernel=pDilationSElem)

        if pCavityFilling:
            for i in range(count):
                temp = cv2.bitwise_not(masks[:, :, i])
                temp, _, _ = cv2.findContours(temp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                masks[:, :, i] = cv2.bitwise_not(temp)
#            for i in range(count):
#                masks[:, :, i] = scipy.ndimage.binary_fill_holes(masks[:, :, i])

        for i in range(count):
            masks[:, :, i] = numpy.where(masks[:, :, i] == 0, 0, 255)

        return kutils.MergeMasks(masks), masks, scores







print("Usage: " + sys.argv[0] + " settings.json")

inputDir = ""
outputDir = ""
modelPath = ""
separate = False
padding = 0.0
dilate = 0
cavityFill = False
confidence = 0.5
scaleFilePath = None
nmsThresh = 0.35
scoreDir = None
defaultSize = None
showOutputs = False

jsn = json.load(open(sys.argv[1]))
params = jsn["segmentation_params"]
inputDir = os.path.join(os.curdir, params["input_dir"])
outputDir = os.path.join(os.curdir, params["output_dir"])
modelPath = os.path.join(os.curdir, params["model"])

if "detection_nms_threshold" in params:
    nmsThresh = float(params["detection_nms_threshold"])
if "default_image_size" in params:
    defaultSize = int(params["default_image_size"])
if "separate_masks" in params:
    separate = params["separate_masks"] == "true"
if "padding" in params:
    padding = float(params["padding"])
if "dilation" in params:
    dilate = int(params["dilation"])
if "cavity_filling" in params:
    cavityFill = params["cavity_filling"] == "true"
if "detection_confidence" in params:
    confidence = float(params["detection_confidence"])
if "scale_file" in params:
    scaleFilePath = params["scale_file"]
if "scores_dir" in params:
    scoreDir = params["scores_dir"]
if "show" in params:
    showOutputs = params["show"] == "true"
if "detection_max_instances" in params:
    maxDetNum=int(params["detection_max_instances"])


imagesDir = os.path.join(inputDir,"images")
imageFiles = [f for f in os.listdir(imagesDir) if os.path.isfile(os.path.join(imagesDir, f))]

method = Segmentation(pModelPath=modelPath, pConfidence=confidence, pNMSThreshold=nmsThresh, pMaxDetNum=maxDetNum)


#parsing scale file if present
scales = {}
if scaleFilePath is not None:
    scaleFile = open(scaleFilePath, "r")
    for line in scaleFile:
        elems = line.split()
        scales[elems[0]] = float(elems[1])
    scaleFile.close()

if len(scales) < 1:
    imageFiles = sorted(imageFiles)

else:
    for imageFile in imageFiles:
        baseName = os.path.splitext(os.path.basename(imageFile))[0]
        if baseName not in scales:
            if defaultSize is not None:
                scales[baseName] = defaultSize
            else:
                print("Missing scaling entry for", baseName, ", skipping")

    #sorting scales
    import operator
    scales = sorted(scales.items(), key=operator.itemgetter(1))
    imageFiles = []
    for file, _ in scales:
        imageFiles.append(file + ".png")

    scales = dict(scales)

os.makedirs(name=outputDir, exist_ok=True)
if scoreDir is not None:
    os.makedirs(name=scoreDir, exist_ok=True)

imcount = len(imageFiles)
for index, imageFile in enumerate(imageFiles):
    print("Image:", str(index + 1), "/", str(imcount), "(", imageFile, ")")

    baseName = os.path.splitext(os.path.basename(imageFile))[0]
    imagePath = os.path.join(imagesDir, imageFile)
    image = skimage.io.imread(imagePath)

    dilationStruct = None
    if dilate > 0:
        dilationStruct = skimage.morphology.disk(dilate)

    maxdim = None
    if baseName in scales:
        maxdim = int(scales[baseName])
    else:
        maxdim = defaultSize

    mask, masks, scores = method.Segment(pImage=image, pPaddingRatio=padding, pCavityFilling=cavityFill, pDilationSElem=dilationStruct, pPredictSize=maxdim)

    count = masks.shape[2]
    print("  Nuclei (including cropped):", str(count))
    if count < 1:
        continue

    skimage.io.imsave(os.path.join(outputDir, baseName + ".tiff"), mask)

    if separate:
        masksDir = os.path.join(outputDir, baseName, "masks")
        os.makedirs(name=masksDir, exist_ok=True)
        for m in range(count):
            skimage.io.imsave(os.path.join(masksDir, str(m) + ".png"), masks[:, :, m])

    if scoreDir is not None:
        scoreFile = open(os.path.join(scoreDir, baseName + ".tsv"),"w")
        scoreFile.write("label\tscore\r\n")
        for s in range(count):
            scoreFile.write(str(s+1) + "\t" + str(scores[s])+ "\r\n")
        scoreFile.close()

    if showOutputs:
        visualize.display_instances(image=image,
                                    boxes=utils.extract_bboxes(masks),
                                    masks=masks,
                                    scores=scores,
                                    title=baseName,
                                    class_ids=numpy.array([1 for _ in range(count)]),
                                    class_names=["BG", "nucleus"]
                                    )