import numpy
import skimage.transform

def convert_image(pImageData, pTargetSize):
    if (len(pImageData.shape) < 3):
        return None
    if (pImageData.shape[2] > 3):
        pImageData = pImageData[:, :, :3]

    norm = 1.0
    type = pImageData.dtype
    if (type == numpy.uint8):
        norm = 1.0 / 256
    elif (type == numpy.uint16):
        norm = 1.0 / 65536
    elif (type == numpy.float):
        norm = 1.0
    else:
        return None

    floatImage = pImageData * norm

    if (floatImage.shape[0] != pTargetSize[0] or floatImage.shape[1] != pTargetSize[1]):
        floatImage = skimage.transform.resize(floatImage, pTargetSize)

    return floatImage