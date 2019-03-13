import sys
import image_augmentation
import matplotlib.pyplot
import matplotlib.image
import skimage.io

print("Usage:", sys.argv[0], "image_path iteration=5")

imagePath = sys.argv[1]
iteration = 5
if len(sys.argv) > 2: iteration = int(sys.argv[2])

image = skimage.io.imread(imagePath)

for i in range(iteration):
    augmap = image_augmentation.GenerateRandomAugmentationMap()
    augmented = image_augmentation.Augment(pImage=image, pAugmentationMap=augmap, pIsMask=False)
    augmentedSize = (augmented.shape[1], augmented.shape[0])
    calculatedSize = image_augmentation.CalculateAugmentedSize(pImage=image, pAugmentationMap=augmap)
    print("real size:", augmentedSize, "\tcalculated size:", str(calculatedSize))
    assert(augmentedSize == calculatedSize)

    matplotlib.pyplot.title(str(i+1) + "/" + str(iteration) + ": " + str(augmap))
    matplotlib.pyplot.imshow(augmented)
    matplotlib.pyplot.show()