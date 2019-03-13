import skimage.transform


def convert_image(image_data):
    # If it has less than 3 channels like gray scale images...
    if len(image_data.shape) < 3:
        return None

    # If RGBA
    if image_data.shape[2] > 3:
        image_data = image_data[:, :, :3]

#    norm = 1.0
#    type = pImageData.dtype
#    if (type == numpy.uint8):
#        norm = 1.0 / 256
#    elif (type == numpy.uint16):
#        norm = 1.0 / 65536
#    elif (type == numpy.float):
#        norm = 1.0
#    else:
#        return None

#    floatImage = pImageData     # * norm

    return image_data


def resize_target_size(image, target_size):
    if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
        image = skimage.transform.resize(image, target_size)

    return image
