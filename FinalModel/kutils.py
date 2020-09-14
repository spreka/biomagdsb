import numpy
import skimage

def RCNNConvertInputImage(pImageData):
    if pImageData.ndim < 2:
        raise ValueError("Invalid image")
    elif pImageData.ndim < 3:
        pImageData = skimage.color.gray2rgb(pImageData)
    if pImageData.shape[2] > 3:
        pImageData = pImageData[:, :, :3]

    # handle 16-bit images
    if pImageData.dtype==numpy.uint16:
        #print('uint16 image, stretching intensities before converting to uint8')
        imagetmp=numpy.zeros((pImageData.shape[0],pImageData.shape[1],3),dtype=numpy.uint8)
        for ch in range(3):
            tmp=pImageData[:,:,ch]
            imgg = tmp.astype(numpy.float)
            tmp=((imgg-numpy.amin(imgg))*255)/(numpy.amax(imgg)-numpy.amin(imgg));
            tmp=tmp.astype(numpy.uint8);
            imagetmp[:,:,ch]=tmp
        pImageData=imagetmp
    else:
        pImageData = pImageData.astype(numpy.uint8)

    return pImageData

def MergeMasks(pMasks):
    if pMasks.ndim < 3:
        raise ValueError("Invalid masks")

    maskCount = pMasks.shape[2]
    width = pMasks.shape[1]
    height = pMasks.shape[0]
    mask = numpy.zeros((height, width), numpy.uint16)

    for i in range(maskCount):
        mask[:,:] = numpy.where(pMasks[:,:,i] != 0, i+1, mask[:,:])

    return mask


def PadImageR(pImageData, pRatio):
    width = pImageData.shape[1]
    height = pImageData.shape[0]

    x = int(float(width) * float(pRatio))
    y = int(float(height) * float(pRatio))

    image = PadImageXY(pImageData, x, y)
    return image, (x, y)


def PadImageXY(pImageData, pX, pY):
    width = pImageData.shape[1]
    height = pImageData.shape[0]

    paddedWidth = width + 2 * pX
    paddedHeight = height + 2 * pY


    if pImageData.ndim > 2:
        count = pImageData.shape[2]
        image = numpy.zeros((paddedHeight, paddedWidth, count), pImageData.dtype)
        for c in range(count):
            image[:, :, c] = numpy.lib.pad(pImageData[:, :, c], ((pY, pY), (pX, pX)), "reflect")

    else:
        image = numpy.lib.pad(pImageData, ((pY, pY), (pX, pX)), "reflect")

    return image