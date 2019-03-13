'''

Image augmentation.

Generates an additional training set from the input training set using some logic. This additional training
set called the augmentated training set will be placed into a different place in the file system with its
own descriptor file together.

Then, the original and the augmented training set will be merged together with the dataset_generator script followed
by the training what takes place in a batched fashion.

'''


import numpy as np

'''
Image augmentation using tiling.
'''


def get_image_tile(image, x, y, xn, yn):
    ys = image.shape[0] // yn
    xs = image.shape[1] // xn
    tile = np.zeros((ys, xs, image.shape[2]))
    for j in range(xs):
        for i in range(ys):
            for k in range(3):
                tile[j, i, k] = image[y*ys+j, x*xs+i, k]
    return tile


def generate_tiles():
    # for x in range(xn):
    #    for y in range(yn):
    #        image_unresized = image_unresized_list[index]
    #        im_tile = get_image_tile(image_unresized, x, y, xn, yn)
    #        im_tile_resized = image_convert.resize_image(im_tile, target_size)

    #        train_image_ids.append("{}_{}_{}".format(image_ids[index], x, y))
    #        train_images.append(im_tile_resized)
    #        train_class_values.append(image_class_values[index])
    pass
