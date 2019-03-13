import os
import numpy as np
from skimage import io
from skimage.morphology import dilation
from skimage.measure import label, regionprops


def get_objects(image, p_threshold=0.5, dilate=0):
    thresholded_image = (image > p_threshold * np.max(image))
    object_image = label(thresholded_image)
    if dilate:
        for _ in range(dilate):
            object_image = dilation(object_image)
    return object_image


if __name__ == '__main__':
    pmap_root = '/media/tdanka/B8703F33703EF828/tdanka/pmaps/'
    seg_folder = '/media/tdanka/B8703F33703EF828/tdanka/pmaps/selected'

    dilate = 3
    label_encoded_path = os.path.join(pmap_root, 'dilate=%d' % dilate)
    os.makedirs(label_encoded_path)
    for image_name in os.listdir(seg_folder):
        image_pmap = io.imread(os.path.join(seg_folder, image_name))
        object_image = get_objects(image_pmap[:, :, 1], p_threshold=0.5, dilate=dilate)
        io.imsave(os.path.join(label_encoded_path, image_name), object_image)

