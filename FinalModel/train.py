'''
import tensorflow

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.Session(config=config)
'''

import sys
import json
import os
import os.path
import numpy
import model
import visualize
import utils
import mask_rcnn_additional
import random

print("Usage", sys.argv[0], "settings.json")

class MaskTrain:
    __mParams = {}

    def __init__(self, pParams):
        self.__mParams = pParams

    def Train(self):
        fixedRandomSeed = None
        trainToValidationChance = 0.2
        includeEvaluationInValidation = True
        stepMultiplier = None
        stepCount = 1000
        showInputs = False
        augmentationLevel = 0
        detNMSThresh = 0.35
        rpnNMSThresh = 0.55
        trainDir = os.path.join(os.curdir, self.__mParams["train_dir"])
        evalDir = None
        inModelPath = os.path.join(os.curdir, self.__mParams["input_model"])
        outModelPath = os.path.join(os.curdir, self.__mParams["output_model"])
        blankInput = self.__mParams["blank_mrcnn"] == "true"
        maxdim = 1024

        if "eval_dir" in self.__mParams:
            evalDir = os.path.join(os.curdir, self.__mParams["eval_dir"])

        if "image_size" in self.__mParams:
            maxdim = int(self.__mParams["image_size"])

        if "train_to_val_seed" in self.__mParams:
            fixedRandomSeed = self.__mParams["train_to_val_seed"]

        if "train_to_val_ratio" in self.__mParams:
            trainToValidationChance = float(self.__mParams["train_to_val_ratio"])

        if "use_eval_in_val" in self.__mParams:
            includeEvaluationInValidation = self.__mParams["use_eval_in_val"] == "true"

        if "step_ratio" in self.__mParams:
            stepMultiplier = float(self.__mParams["step_ratio"])

        if "step_num" in self.__mParams:
            stepCount = int(self.__mParams["step_num"])

        if "show_inputs" in self.__mParams:
            showInputs = self.__mParams["show_inputs"] == "true"

        if "random_augmentation_level" in self.__mParams:
            augmentationLevel = int(self.__mParams["random_augmentation_level"])

        if "detection_nms_threshold" in self.__mParams:
            detNMSThresh = float(self.__mParams["detection_nms_threshold"])

        if "rpn_nms_threshold" in self.__mParams:
            rpnNMSThresh = float(self.__mParams["rpn_nms_threshold"])



        rnd = random.Random()
        rnd.seed(fixedRandomSeed)
        trainImagesAndMasks = {}
        validationImagesAndMasks = {}

        # iterate through train set
        imagesDir = os.path.join(trainDir, "images")
        masksDir = os.path.join(trainDir, "masks")

        # splitting train data into train and validation
        imageFileList = [f for f in os.listdir(imagesDir) if os.path.isfile(os.path.join(imagesDir, f))]
        for imageFile in imageFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            imagePath = os.path.join(imagesDir, imageFile)
            maskPath = os.path.join(masksDir, baseName + ".tiff")
            if not os.path.isfile(imagePath) or not os.path.isfile(maskPath):
                continue
            if rnd.random() > trainToValidationChance:
                trainImagesAndMasks[imagePath] = maskPath
            else:
                validationImagesAndMasks[imagePath] = maskPath

        # adding evaluation data into validation
        if includeEvaluationInValidation and evalDir is not None:

            # iterate through test set
            imagesDir = os.path.join(evalDir, "images")
            masksDir = os.path.join(evalDir, "masks")

            imageFileList = [f for f in os.listdir(imagesDir) if os.path.isfile(os.path.join(imagesDir, f))]
            for imageFile in imageFileList:
                baseName = os.path.splitext(os.path.basename(imageFile))[0]
                imagePath = os.path.join(imagesDir, imageFile)
                maskPath = os.path.join(masksDir, baseName + ".tiff")
                if not os.path.isfile(imagePath) or not os.path.isfile(maskPath):
                    continue
                validationImagesAndMasks[imagePath] = maskPath

        if len(trainImagesAndMasks) < 1:
            raise ValueError("Empty train image list")

        #just to be non-empty
        if len(validationImagesAndMasks) < 1:
            for key, value in trainImagesAndMasks.items():
                validationImagesAndMasks[key] = value
                break

        # Training dataset
        dataset_train = mask_rcnn_additional.NucleiDataset()
        dataset_train.initialize(pImagesAndMasks=trainImagesAndMasks, pAugmentationLevel=augmentationLevel)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = mask_rcnn_additional.NucleiDataset()
        dataset_val.initialize(pImagesAndMasks=validationImagesAndMasks, pAugmentationLevel=0)
        dataset_val.prepare()

        print("training images (with augmentation):", dataset_train.num_images)
        print("validation images (with augmentation):", dataset_val.num_images)

        config = mask_rcnn_additional.NucleiConfig()
        config.IMAGE_MAX_DIM = maxdim
        config.IMAGE_MIN_DIM = maxdim
        config.STEPS_PER_EPOCH = stepCount
        if stepMultiplier is not None:
            steps = int(float(dataset_train.num_images) * stepMultiplier)
            config.STEPS_PER_EPOCH = steps

        config.VALIDATION_STEPS = dataset_val.num_images
        config.DETECTION_NMS_THRESHOLD = detNMSThresh
        config.RPN_NMS_THRESHOLD = rpnNMSThresh
        config.__init__()
        # show config
        config.display()

        # show setup
        for a in dir(self):
            if not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

        if showInputs:
            # Load and display random samples
            image_ids = numpy.random.choice(dataset_train.image_ids, 20)
            for imageId in image_ids:
                image = dataset_train.load_image(imageId)
                mask, class_ids = dataset_train.load_mask(imageId)
                # visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
                visualize.display_instances(image=image, masks=mask, class_ids=class_ids,
                                            title=dataset_train.image_reference(imageId),
                                            boxes=utils.extract_bboxes(mask), class_names=dataset_train.class_names)

        # Create model in training mode
        mdl = model.MaskRCNN(mode="training", config=config, model_dir=os.path.dirname(outModelPath))

        if blankInput:
            mdl.load_weights(inModelPath, by_name=True,
                             exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        else:
            mdl.load_weights(inModelPath, by_name=True)

        allcount = 0
        for epochgroup in self.__mParams["epoch_groups"]:
            epochs = int(epochgroup["epochs"])
            if epochs < 1:
                continue
            allcount += epochs
            mdl.train(dataset_train,
                      dataset_val,
                      learning_rate=float(epochgroup["learning_rate"]),
                      epochs=allcount,
                      layers=epochgroup["layers"])

        mdl.keras_model.save_weights(outModelPath)


jsn = json.load(open(sys.argv[1]))
trainer = MaskTrain(pParams=jsn["train_params"])
trainer.Train()


