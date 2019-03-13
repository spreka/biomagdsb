import numpy
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps
import random
import copy
import imgaug
import skimage.io

'''
@brief Augments an image or mask
@param pImage image as numpy array or PIL image or image path
@param pAugmentationMap is a map of augmentation parameters, see below
@param pIsMask is the image a binary or multi label mask (different interpolation is used for masks)
@return image as numpy array

augmentation map can contain the following parameters and this is the defined order of the execution
crop: tuple of x,y,width,height
scale_x, scale_y: positive float
flip_x, flip_y: (True or False)
rotation: int (90, 180 or 270)
brightness, contrast, sharpness: float (0.0-2.0)
noise: float (0.0-1.0) noise strength. 1.0 means that the maximum deviation from a subpixel is the maximum brightness of the image in either positive or negative direction
invert: (True or False)
color_shift: tuple of three floats (-1.0-1.0) plus/minus value for that channel
'''
def Augment(pImage, pAugmentationMap, pIsMask = False):
    if isinstance(pImage, PIL.Image.Image):
        image = numpy.array(pImage)
    elif isinstance(pImage, str):
        image = skimage.io.imread(pImage)
    elif isinstance(pImage, (numpy.ndarray, numpy.generic)):
        image = copy.deepcopy(pImage)

    interpolation = PIL.Image.BICUBIC
    if pIsMask:
        interpolation = PIL.Image.NEAREST

    if "crop" in pAugmentationMap:
        (xf, yf, widthf, heightf) = pAugmentationMap["crop"]
        width = int(float(image.shape[1]) * widthf)
        height = int(float(image.shape[0]) * heightf)
        x = int(float(image.shape[1]) * xf)
        y = int(float(image.shape[0]) * yf)
        if image.ndim == 2:
            image = image[y:y+height,x:x+width]
        else:
            image = image[y:y + height, x:x + width,:]

    if "scale_x" in pAugmentationMap:
        scale_x = float(pAugmentationMap["scale_x"])
        pil = PIL.Image.fromarray(image)
        pil = pil.resize(size=(int(float(pil.size[0]) * scale_x), pil.size[1]), resample=interpolation)
        image = numpy.array(pil)

    if "scale_y" in pAugmentationMap:
        scale_y = float(pAugmentationMap["scale_y"])
        pil = PIL.Image.fromarray(image)
        pil = pil.resize(size=(pil.size[0],int(float(pil.size[1]) * scale_y)), resample=interpolation)
        image = numpy.array(pil)


    if "flip_x" in pAugmentationMap and pAugmentationMap["flip_x"]:
        image = numpy.fliplr(image)

    if "flip_y" in pAugmentationMap and pAugmentationMap["flip_y"]:
        image = numpy.flipud(image)

    if "rotation" in pAugmentationMap:
        rotation = int(pAugmentationMap["rotation"])
        if rotation == 90:
            image = numpy.rot90(image)
        elif rotation == 180:
            image = numpy.rot90(image)
            image = numpy.rot90(image)
        elif rotation == 270:
            image = numpy.rot90(image)
            image = numpy.rot90(image)
            image = numpy.rot90(image)

    if pIsMask:
        return image

    else:
        pil = None

        if "brightness" in pAugmentationMap:
            if pil is None: pil = PIL.Image.fromarray(image)
            factor = float(pAugmentationMap["brightness"])
            enhancer = PIL.ImageEnhance.Brightness(pil)
            pil = enhancer.enhance(factor=factor)

        if "contrast" in pAugmentationMap:
            if pil is None: pil = PIL.Image.fromarray(image)
            factor = float(pAugmentationMap["contrast"])
            enhancer = PIL.ImageEnhance.Contrast(pil)
            pil = enhancer.enhance(factor=factor)

        if "sharpness" in pAugmentationMap:
            if pil is None: pil = PIL.Image.fromarray(image)
            factor = float(pAugmentationMap["sharpness"])
            enhancer = PIL.ImageEnhance.Sharpness(pil)
            pil = enhancer.enhance(factor=factor)

        if pil is not None:
            if pil.mode != "RGB":
                pil = image.convert(mode="RGB")
            image = numpy.array(pil, dtype=numpy.uint8)

        if "noise" in pAugmentationMap:
            strength = float(pAugmentationMap["noise"])
            maximum = float(image.max())
            noise = numpy.random.random(image.shape)    # noise is now in 0 - 1 range
            noise *= maximum * 2.0                      # noise is now in 0 - ~512 range
            noise = maximum - noise                     # noise is now in the ~-255, ~255 range
            noise *= strength                           # noise is weighted down by strength
            image = (image.astype("float") + noise).clip(0,255).astype("uint8")

        if "invert" in pAugmentationMap and pAugmentationMap["invert"]:
            if image.ndim > 2:
                image[:,:,:3] = 255 - image[:,:,:3]
            else:
                image = 255 - image

        if "color_shift" in pAugmentationMap and image.ndim > 2:
            shift = pAugmentationMap["color_shift"]
            r = int(255.0 * float(shift[0]))
            g = int(255.0 * float(shift[1]))
            b = int(255.0 * float(shift[2]))
            intArr = image.astype("int")
            intArr[:,:,0] += r
            intArr[:,:,1] += g
            intArr[:,:,2] += b
            image = intArr.clip(0, 255).astype("uint8")

        return image

'''
@brief does not augment but calculates the dimensions
@param pImage image as numpy array or PIL image or image path
@param pAugmentationMap is a map of augmentation parameters, see over
@return new size in width, height format
'''
def CalculateAugmentedSize(pImage, pAugmentationMap):
    width = 0
    height = 0
    if isinstance(pImage, PIL.Image.Image):
        width = pImage.size[0]
        height = pImage.size[1]
    elif isinstance(pImage, str):
        image = PIL.Image.open(pImage, "r")
        width = image.size[0]
        height = image.size[1]
    elif isinstance(pImage, (numpy.ndarray, numpy.generic)):
        width = pImage.shape[1]
        height = pImage.shape[0]

    if "crop" in pAugmentationMap:
        (_, _, widthf, heightf) = pAugmentationMap["crop"]
        width = int(float(width) * widthf)
        height = int(float(height) * heightf)

    if "scale_x" in pAugmentationMap:
        scale_x = float(pAugmentationMap["scale_x"])
        width = int(float(width) * scale_x)

    if "scale_y" in pAugmentationMap:
        scale_y = float(pAugmentationMap["scale_y"])
        height = int(float(height) * scale_y)

    if "rotation" in pAugmentationMap:
        rotation = int(pAugmentationMap["rotation"])
        if rotation == 90 or rotation == 270:
            width, height = height, width

    return width, height

def GenerateRandomAugmentationMap(
        pSeed = None,               # fixed seed for the random generator
        pEnableCropping = True,     # enable cropping
        pCropFactor = 0.5,          # maximum crop out size (minimum new size is 1.0-pCropFactor)
        pEnableScaling = True,      # enable x,y scaling
        pScaleFactor = 0.15,        # maximum scale factor
        pEnableFlipping = True,     # enable x,y flippinh
        pEnableRotation = True,     # enable simple rotation
        pEnableBrightness = True,   # enable brightness change
        pBrightnessFactor = 0.20,   # maximum +- brightness
        pEnableContrast = True,     # enable contrast change
        pContrastFactor = 0.20,     # maximum +- contrast
        pEnableSharpness = True,    # enable softness/sharpness change
        pSharpnessFactor = 0.20,    # maximum +- sharpness
        pEnableNoise = True,        # enable random noise
        pMaxNoiseStrength = 0.10,   # random noise strength (maximum deviation from the original subpixel brightness by the weighted max brightness of the image)
        pEnableInvert = True,       # enable color inversion
        pEnableColorShift = True,   # enable color shift
        pColorShiftStrength = 0.25  # maximum +- color shift
):
    augmentationMap = {}
    rnd = random.Random()
    rnd.seed(pSeed)

    def FactoredRandomFloat(pFactor):
        factor = int(pFactor * 100.0)
        return float(rnd.randint(100-factor,100+factor)) / 100.0

    def ShouldDo():
        return rnd.randint(0, 1) == 1

    if pEnableCropping and ShouldDo():
        width = 1.0 - (rnd.random() * pCropFactor)
        height = 1.0 - (rnd.random() * pCropFactor)
        x = (1.0 - width) * rnd.random()
        y = (1.0 - height) * rnd.random()
        augmentationMap["crop"] = (x, y, width, height)


    if pEnableScaling:
        if ShouldDo():
            augmentationMap["scale_x"] = FactoredRandomFloat(pScaleFactor)
        if ShouldDo():
            augmentationMap["scale_y"] = FactoredRandomFloat(pScaleFactor)

    if pEnableFlipping:
        if ShouldDo():
            augmentationMap["flip_x"] = True

        if ShouldDo():
            augmentationMap["flip_y"] = True

    if pEnableRotation and ShouldDo():
        rotation = rnd.randint(1, 3)
        if rotation == 1:
            augmentationMap["rotation"] = 90
        elif rotation == 2:
            augmentationMap["rotation"] = 180
        elif rotation == 3:
            augmentationMap["rotation"] = 270

    if pEnableBrightness and ShouldDo():
        augmentationMap["brightness"] = FactoredRandomFloat(pBrightnessFactor)

    if pEnableContrast and ShouldDo():
        augmentationMap["contrast"] = FactoredRandomFloat(pContrastFactor)

    if pEnableSharpness and ShouldDo():
        augmentationMap["sharpness"] = FactoredRandomFloat(pSharpnessFactor)

    if pEnableNoise and pMaxNoiseStrength > 0 and ShouldDo():
        augmentationMap["noise"] = rnd.random() * pMaxNoiseStrength

    if pEnableInvert and ShouldDo():
        augmentationMap["invert"] = True

    if pEnableColorShift and ShouldDo():
        r = pColorShiftStrength - (2.0 * rnd.random() * pColorShiftStrength)
        g = pColorShiftStrength - (2.0 * rnd.random() * pColorShiftStrength)
        b = pColorShiftStrength - (2.0 * rnd.random() * pColorShiftStrength)
        augmentationMap["color_shift"] = (r,g,b)

    return augmentationMap


#!!!Important: If you wish to use scaling, make sure that the following are added to Mask_RCNN's MASK_AUGMENTERS list: "Scale"
def GenerateRandomImgaugAugmentation(
        pSeed=None,                     # random seed
        pAugmentationLevel=5,           # maximum number of augmentations per image
        pEnableScaling=False,           # enable x,y scaling
        pScaleFactor=0.15,              # maximum scale factor
        pEnableCropping=True,           # enable cropping
        pCropFactor=0.5,                # maximum crop out size (minimum new size is 1.0-pCropFactor)
        pEnableFlipping=True,           # enable x,y flipping
        pEnableBlur=True,               # enable gaussian blur
        pBlurSigma=1.0,                 # maximum sigma of gaussian blur
        pEnableSharpness=True,          # enable sharpness change
        pSharpnessFactor=0.5,           # maximum additional sharpness
        pEnableEmboss=True,             # enable emboss change
        pEmbossFactor=0.50,             # maximum emboss
        pEnableColorShift=True,         # enable color shift change
        pColorShiftFactor=0.20,         # maximum +- brightness per channel
        pEnableBrightness=True,         # enable brightness change
        pBrightnessFactor=0.20,         # maximum +- brightness
        pEnableGuassianNoise=True,      # enable additive gaussian noise
        pMaxGuassianNoise=0.10,         # maximum gaussian noise strength
        pEnableRandomNoise=True,        # enable random noise
        pMaxRandomNoise=0.10,           # maximum random noise strength
        pEnableDropOut=True,            # enable pixel dropout
        pMaxDropoutPercentage=0.2,      # maximum dropout percentage
        pEnableInvert=True,             # enables color invert
        pEnableContrast=True,           # enable contrast change
        pContrastFactor=0.20,           # maximum +- contrast
        pEnableRotation=True,           # enable rotation
        pEnableShear=True,              # enable shear
        pMaxShearDegree=30,             # maximum shear degree
        pEnablePiecewiseAffine=True,    # enable piecewise affine transformation
        pPiecewiseAffineStrength=0.05,  # maximum piecewise affine strength
        pMode="constant"                # affine fill mode, do not use anything else
):
    if pAugmentationLevel < 1:
        return None

    imgaug.seed(pSeed)
    augmentationMap = []

    augmentationMap.append(imgaug.augmenters.Noop(deterministic=True))

    if pEnableScaling:
        aug = imgaug.augmenters.Scale(size=({"height":(1.0 - pScaleFactor, 1.0), "width":(1.0 - pScaleFactor, 1.0)}), deterministic=True)
        augmentationMap.append(aug)

    if pEnableCropping:
        maxcrop = pCropFactor * 0.5
        aug = imgaug.augmenters.Crop(percent=((0.0,maxcrop), (0.0,maxcrop), (0.0,maxcrop), (0.0,maxcrop)),deterministic=True)
        augmentationMap.append(aug)

    if pEnableFlipping:
        aug = imgaug.augmenters.Fliplr(deterministic=True)
        augmentationMap.append(aug)
        aug = imgaug.augmenters.Flipud(deterministic=True)
        augmentationMap.append(aug)

    if pEnableBlur:
        aug = imgaug.augmenters.GaussianBlur((0.0, pBlurSigma),deterministic=True)
        augmentationMap.append(aug)

    if pEnableSharpness:
        aug = imgaug.augmenters.Sharpen((0.0, pSharpnessFactor),deterministic=True)
        augmentationMap.append(aug)

    if pEnableEmboss:
        aug = imgaug.augmenters.Emboss((0.0, pEmbossFactor),deterministic=True)
        augmentationMap.append(aug)

    if pEnableBrightness:
        brightness = int(255.0 * pBrightnessFactor)
        aug = imgaug.augmenters.Add((-brightness, brightness),deterministic=True)
        augmentationMap.append(aug)

    if pEnableColorShift:
        brightness = int(255.0 * pColorShiftFactor)
        aug = imgaug.augmenters.Add((-brightness, brightness),per_channel=True,deterministic=True)
        augmentationMap.append(aug)

    if pEnableGuassianNoise:
        noise = 255.0 * pMaxGuassianNoise
        aug = imgaug.augmenters.AdditiveGaussianNoise(scale=(0.0, noise),deterministic=True)
        augmentationMap.append(aug)

    if pEnableRandomNoise:
        aug = imgaug.augmenters.MultiplyElementwise((1.0-pMaxRandomNoise, 1.0+pMaxGuassianNoise), per_channel=True,deterministic=True)
        augmentationMap.append(aug)

    if pEnableDropOut:
        aug = imgaug.augmenters.Dropout(p=(0.0, pMaxDropoutPercentage), per_channel=False,deterministic=True)
        augmentationMap.append(aug)

    if pEnableInvert:
        aug = imgaug.augmenters.Invert(0.5,deterministic=True)
        augmentationMap.append(aug)

    if pEnableContrast:
        aug = imgaug.augmenters.ContrastNormalization((1.0-pContrastFactor, 1.0+pContrastFactor),deterministic=True)
        augmentationMap.append(aug)

    if pEnableRotation:
        aug = imgaug.augmenters.Affine(rotate=(0.0, 360.0), mode=pMode,deterministic=True)
        augmentationMap.append(aug)

    if pEnableShear:
        aug = imgaug.augmenters.Affine(shear=(-pMaxShearDegree, pMaxShearDegree), mode=pMode,deterministic=True)
        augmentationMap.append(aug)

    if pEnablePiecewiseAffine:
        aug = imgaug.augmenters.PiecewiseAffine(scale=(0.0, pPiecewiseAffineStrength),deterministic=True)
        augmentationMap.append(aug)

    return imgaug.augmenters.SomeOf((0,pAugmentationLevel), augmentationMap, random_order=True,deterministic=True)
